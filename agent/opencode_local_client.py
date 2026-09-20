"""OpenAI-compatible shim that drives a local ``opencode serve`` subprocess over HTTP.

``opencode-local`` spawns and owns one long-lived ``opencode serve`` process (lazy start,
health-checked, cleaned up at exit) instead of hitting OpenCode's hosted Zen/Go relay directly.
The relay 403s anonymous access from anything but its own client (see the unknown-provider hint
in ``hermes_cli/auth.py``); the local server IS that official client, so it carries its own
trusted session instead of an ``OPENCODE_*_API_KEY``.

Unlike Copilot ACP (a short-lived session per request), ``opencode serve`` is meant to run as a
persistent local server, so the subprocess is a process-wide singleton reused across requests
rather than spawned per call.

The local server mirrors the hosted relay's per-model wire dialect (``_OPENCODE_API_MODE_PREFIXES``
in ``hermes_cli/models.py``): Muse Spark / GPT / Grok speak the Responses API, Claude / MiniMax /
Qwen speak the Messages API, everything else speaks Chat Completions. Picking the right dialect per
model is what makes reasoning arrive as structured thinking and tool calls arrive as structured
events instead of prose — the wrong dialect is the actual cause of both failure modes, not
something that needs a separate text-scraping workaround (contrast ``acp_openai_bridge.py``, which
DOES need one because ACP has no tools/tool_calls channel at all).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import build_openai_tool_call
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

OPENCODE_LOCAL_MARKER_BASE_URL = "opencode-local://127.0.0.1"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_HEALTH_CHECK_TIMEOUT_SECONDS = 20.0
_HEALTH_CHECK_INTERVAL_SECONDS = 0.2

# Per-model wire dialect the local server speaks, mirroring ``_OPENCODE_API_MODE_PREFIXES``
# (hermes_cli/models.py) for the hosted Zen/Go relay. Checked in order; first match wins.
_API_MODE_PREFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("muse-spark", "gpt-", "grok-"), "codex_responses"),
    (("claude-", "minimax-", "qwen"), "anthropic_messages"),
)


def opencode_local_model_api_mode(model: str | None) -> str:
    """Wire dialect ``opencode serve`` speaks locally for *model* (see ``_API_MODE_PREFIXES``).

    A single fixed dialect is wrong for all but one model family: Claude's extended-thinking
    blocks and OpenAI Responses reasoning items only round-trip correctly on their native
    endpoints, so every request is routed per-model rather than per-provider.
    """
    normalized = str(model or "").strip().lower()
    for prefixes, mode in _API_MODE_PREFIXES:
        if normalized.startswith(prefixes):
            return mode
    return "chat_completions"


# ── Subprocess lifecycle ─────────────────────────────────────────────────────────────────────


def _resolve_command() -> str:
    return os.getenv("HERMES_OPENCODE_LOCAL_COMMAND", "").strip() or "opencode"


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_OPENCODE_LOCAL_ARGS", "").strip()
    return shlex.split(raw) if raw else ["serve"]


def _resolve_configured_port() -> int:
    """Explicit ``HERMES_OPENCODE_LOCAL_PORT``, else 0 (= pick a free ephemeral port ourselves
    rather than guessing the CLI's own default, which may already be taken by another instance)."""
    raw = os.getenv("HERMES_OPENCODE_LOCAL_PORT", "").strip()
    if raw:
        try:
            port = int(raw)
            if 0 < port < 65536:
                return port
        except ValueError:
            pass
    return 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_external_base_url() -> str:
    """An operator-managed ``opencode serve`` to reuse instead of spawning our own — avoids the
    port pick / health-check race and orphan risk entirely when one is already running."""
    return os.getenv("HERMES_OPENCODE_LOCAL_URL", "").strip().rstrip("/")


def _build_subprocess_env() -> dict[str, str]:
    # opencode drives its own LLM provider credentials (its own auth.json / API keys), so it
    # needs the same credential-inheriting env as Copilot ACP. See agent/copilot_acp_client.py.
    return hermes_subprocess_env(inherit_credentials=True)


def _windows_hide_flags() -> int:
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags

        return windows_hide_flags()
    except Exception:
        return 0


class _OpencodeLocalServer:
    """One lazily-spawned, health-checked ``opencode serve`` process, shared by every client
    instance in this process that was built with the same command/args. A second caller racing
    the first spawn blocks on the same lock instead of starting a duplicate process."""

    _instances: dict[tuple[str, ...], "_OpencodeLocalServer"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, command: str, args: list[str]):
        self._command = command
        self._args = args
        self._proc: subprocess.Popen | None = None
        self._base_url = ""
        self._external = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls, command: str, args: list[str]) -> "_OpencodeLocalServer":
        key = (command, *args)
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls._instances[key] = cls(command, list(args))
            return inst

    def base_url(self, *, timeout_seconds: float) -> str:
        with self._lock:
            if self._base_url:
                return self._base_url
            if external := _resolve_external_base_url():
                self._base_url = external
                self._external = True
                return self._base_url
            self._spawn_and_wait(timeout_seconds)
            return self._base_url

    def _spawn_and_wait(self, timeout_seconds: float) -> None:
        port = _resolve_configured_port() or _free_port()
        argv = [self._command, *self._args, "--port", str(port), "--hostname", "127.0.0.1"]
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", env=_build_subprocess_env(),
                creationflags=_windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start opencode-local command '{self._command}'. Install the OpenCode "
                "CLI (https://opencode.ai) or set HERMES_OPENCODE_LOCAL_COMMAND to its path."
            ) from exc
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr_text = (proc.stderr.read() if proc.stderr else "").strip()
                raise RuntimeError(
                    f"opencode serve exited during startup (code {proc.returncode}): "
                    f"{stderr_text or '(no stderr)'}"
                )
            try:
                with urllib.request.urlopen(base_url + "/", timeout=1.0) as resp:
                    resp.read(1)
                healthy = True
                break
            except Exception as exc:  # server not listening yet, or a 4xx/5xx root — both mean "alive enough"
                if isinstance(exc, urllib.error.HTTPError):
                    healthy = True
                    break
                last_error = exc
                time.sleep(_HEALTH_CHECK_INTERVAL_SECONDS)
        if not healthy:
            with contextlib.suppress(Exception):
                proc.terminate()
            raise TimeoutError(f"opencode serve did not become healthy within {timeout_seconds:.0f}s: {last_error}")
        self._proc = proc
        self._base_url = base_url
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()

    def close(self) -> None:
        if not self._external:
            self._shutdown()


# ── Wire-shape conversion (OpenAI messages -> per-dialect payload) ─────────────────────────────


def _system_and_rest(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if (m.get("role") or "").strip().lower() == "system":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
        else:
            rest.append(m)
    return "\n\n".join(system_parts), rest


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
        return "\n".join(p for p in parts if p)
    return ""


def _to_chat_payload(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    tool_choice: Any, stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def _to_responses_payload(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    tool_choice: Any, stream: bool,
) -> dict[str, Any]:
    instructions, rest = _system_and_rest(messages)
    input_items: list[dict[str, Any]] = []
    for m in rest:
        role = (m.get("role") or "user").strip().lower()
        if role == "tool":
            input_items.append({
                "type": "function_call_output", "call_id": m.get("tool_call_id") or "",
                "output": _text_of(m.get("content")),
            })
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            input_items.append({
                "type": "function_call", "call_id": call.get("id") or "", "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "{}",
            })
        text = _text_of(m.get("content"))
        if text:
            input_items.append({
                "type": "message", "role": "assistant" if role == "assistant" else "user",
                "content": [{"type": "output_text" if role == "assistant" else "input_text", "text": text}],
            })
    payload: dict[str, Any] = {"model": model, "input": input_items, "stream": stream}
    if instructions:
        payload["instructions"] = instructions
    if tools:
        payload["tools"] = [
            {"type": "function", "name": (t.get("function") or {}).get("name"),
             "description": (t.get("function") or {}).get("description", ""),
             "parameters": (t.get("function") or {}).get("parameters", {})}
            for t in tools if isinstance(t, dict) and t.get("function")
        ]
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


_ANTHROPIC_TOOL_CHOICE = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": None}


def _to_anthropic_payload(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
    tool_choice: Any, stream: bool, *, max_tokens: int,
) -> dict[str, Any]:
    system, rest = _system_and_rest(messages)
    anthropic_messages: list[dict[str, Any]] = []
    for m in rest:
        role = (m.get("role") or "user").strip().lower()
        if role == "tool":
            anthropic_messages.append({
                "role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": m.get("tool_call_id") or "",
                    "content": _text_of(m.get("content")),
                }],
            })
            continue
        blocks: list[dict[str, Any]] = []
        text = _text_of(m.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except Exception:
                arguments = {}
            blocks.append({"type": "tool_use", "id": call.get("id") or "", "name": fn.get("name") or "", "input": arguments})
        if blocks:
            anthropic_messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks})
    payload: dict[str, Any] = {
        "model": model, "messages": anthropic_messages, "max_tokens": max_tokens, "stream": stream,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = [
            {"name": (t.get("function") or {}).get("name"), "description": (t.get("function") or {}).get("description", ""),
             "input_schema": (t.get("function") or {}).get("parameters", {})}
            for t in tools if isinstance(t, dict) and t.get("function")
        ]
    if isinstance(tool_choice, str) and tool_choice in _ANTHROPIC_TOOL_CHOICE:
        choice = _ANTHROPIC_TOOL_CHOICE[tool_choice]
        if choice is not None:
            payload["tool_choice"] = choice
        else:
            payload.pop("tools", None)
    return payload


# ── Response parsing (per-dialect JSON -> uniform text/reasoning/tool_calls) ───────────────────


def _parse_chat_response(data: dict[str, Any]) -> tuple[str, str, list[Any], str]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = [
        build_openai_tool_call(
            call_id=tc.get("id") or f"call_{i}", name=(tc.get("function") or {}).get("name") or "",
            arguments=(tc.get("function") or {}).get("arguments") or "{}",
        )
        for i, tc in enumerate(message.get("tool_calls") or [])
    ]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
    return message.get("content") or "", reasoning, tool_calls, finish_reason


def _parse_responses_response(data: dict[str, Any]) -> tuple[str, str, list[Any], str]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[Any] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text") or "")
        elif item_type == "reasoning":
            for part in item.get("summary") or item.get("content") or []:
                if isinstance(part, dict):
                    reasoning_parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    reasoning_parts.append(part)
        elif item_type == "function_call":
            tool_calls.append(build_openai_tool_call(
                call_id=item.get("call_id") or item.get("id") or f"call_{len(tool_calls)}",
                name=item.get("name") or "", arguments=item.get("arguments") or "{}",
            ))
    finish_reason = "tool_calls" if tool_calls else "stop"
    return "".join(text_parts), "".join(reasoning_parts), tool_calls, finish_reason


def _parse_anthropic_response(data: dict[str, Any]) -> tuple[str, str, list[Any], str]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[Any] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text") or "")
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking") or "")
        elif block_type == "tool_use":
            tool_calls.append(build_openai_tool_call(
                call_id=block.get("id") or f"call_{len(tool_calls)}", name=block.get("name") or "",
                arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
            ))
    stop_reason = data.get("stop_reason")
    finish_reason = "tool_calls" if tool_calls else ("length" if stop_reason == "max_tokens" else "stop")
    return "".join(text_parts), "".join(reasoning_parts), tool_calls, finish_reason


_ENDPOINTS = {
    "chat_completions": "/v1/chat/completions",
    "codex_responses": "/v1/responses",
    "anthropic_messages": "/v1/messages",
}


def _effective_timeout(timeout: Any) -> float:
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    return max((float(v) for v in candidates if isinstance(v, (int, float))), default=_DEFAULT_TIMEOUT_SECONDS)


# ── SSE plumbing ─────────────────────────────────────────────────────────────────────────────


def _iter_sse_events(resp: Any) -> Iterator[dict[str, Any]]:
    """Yield each ``data:`` JSON payload from an SSE stream, forwarding lines/chunks as they
    arrive rather than buffering the whole body first (streamed text must keep its own line
    breaks intact, not get re-flowed through a batch-then-rechunk pass)."""
    data_lines: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except Exception:
                    continue
            continue
        if line.startswith(":"):
            continue  # SSE comment / keep-alive ping
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            with contextlib.suppress(Exception):
                yield json.loads(payload)


def _chat_stream_delta(event: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]] | None, str | None]:
    choice = (event.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    tool_calls = delta.get("tool_calls")
    return delta.get("content") or "", delta.get("reasoning_content") or delta.get("reasoning") or "", tool_calls, choice.get("finish_reason")


def _responses_stream_delta(event: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]] | None, str | None]:
    event_type = event.get("type") or ""
    if event_type == "response.output_text.delta":
        return event.get("delta") or "", "", None, None
    if event_type == "response.reasoning_summary_text.delta":
        return "", event.get("delta") or "", None, None
    if event_type == "response.completed":
        return "", "", None, "stop"
    return "", "", None, None


def _anthropic_stream_delta(event: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]] | None, str | None]:
    event_type = event.get("type") or ""
    if event_type == "content_block_delta":
        delta = event.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            return delta.get("text") or "", "", None, None
        if delta_type == "thinking_delta":
            return "", delta.get("thinking") or "", None, None
        if delta_type == "input_json_delta":
            return "", "", [{"index": event.get("index", 0), "id": None, "type": "function",
                              "function": {"name": None, "arguments": delta.get("partial_json") or ""}}], None
    if event_type == "content_block_start":
        block = event.get("content_block") or {}
        if block.get("type") == "tool_use":
            return "", "", [{"index": event.get("index", 0), "id": block.get("id"), "type": "function",
                              "function": {"name": block.get("name"), "arguments": ""}}], None
    if event_type == "message_delta":
        stop_reason = (event.get("delta") or {}).get("stop_reason")
        if stop_reason:
            return "", "", None, "tool_calls" if stop_reason == "tool_use" else "stop"
    return "", "", None, None


_STREAM_DELTA_PARSERS = {
    "chat_completions": _chat_stream_delta, "codex_responses": _responses_stream_delta,
    "anthropic_messages": _anthropic_stream_delta,
}


class OpencodeLocalClient:
    """Minimal OpenAI-client-compatible facade for a local ``opencode serve``."""

    # This shim already owns HTTP I/O end-to-end (subprocess + wire calls), so — like the ACP
    # client — it must not be re-dispatched through another wire adapter.
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, default_headers: dict[str, str] | None = None,
        command: str | None = None, args: list[str] | None = None, **_: Any,
    ):
        self.api_key = api_key or "opencode-local"
        self.base_url = base_url or OPENCODE_LOCAL_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._server = _OpencodeLocalServer.get(self._command, self._args)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion))
        self.is_closed = False

    def close(self) -> None:
        self._server.close()
        self.is_closed = True

    def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
        base = self._server.base_url(timeout_seconds=timeout_seconds)
        req = urllib.request.Request(base.rstrip("/") + "/v1/models")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode())
        items = data if isinstance(data, list) else data.get("data", [])
        return [m["id"] for m in items if isinstance(m, dict) and "id" in m]

    def _post(self, base: str, path: str, payload: dict[str, Any], *, timeout: float, stream: bool) -> Any:
        req = urllib.request.Request(
            base.rstrip("/") + path, data=json.dumps(payload).encode("utf-8"), method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream" if stream else "application/json")
        for k, v in self._default_headers.items():
            req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=timeout)

    def _build_payload(self, mode: str, model: str, messages: list[dict[str, Any]], tools, tool_choice, stream: bool) -> dict[str, Any]:
        if mode == "codex_responses":
            return _to_responses_payload(model, messages, tools, tool_choice, stream)
        if mode == "anthropic_messages":
            return _to_anthropic_payload(model, messages, tools, tool_choice, stream, max_tokens=8192)
        return _to_chat_payload(model, messages, tools, tool_choice, stream)

    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None, timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None, stream: bool = False, **_: Any,
    ) -> Any:
        mode = opencode_local_model_api_mode(model)
        request_timeout = _effective_timeout(timeout)
        base = self._server.base_url(timeout_seconds=max(_HEALTH_CHECK_TIMEOUT_SECONDS, min(request_timeout, 120.0)))
        payload = self._build_payload(mode, model or "", messages or [], tools, tool_choice, stream)
        if stream:
            return self._stream_completion(base, mode, model, payload, request_timeout)
        return self._one_shot_completion(base, mode, model, payload, request_timeout)

    def _one_shot_completion(self, base: str, mode: str, model: str | None, payload: dict[str, Any], timeout: float) -> Any:
        with self._post(base, _ENDPOINTS[mode], payload, timeout=timeout, stream=False) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parser = {
            "chat_completions": _parse_chat_response, "codex_responses": _parse_responses_response,
            "anthropic_messages": _parse_anthropic_response,
        }[mode]
        content, reasoning, tool_calls, finish_reason = parser(data)
        message = SimpleNamespace(
            content=content or None, tool_calls=tool_calls or None, reasoning=reasoning or None,
            reasoning_content=reasoning or None, reasoning_details=None,
        )
        usage = data.get("usage") or {}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(
                prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
                prompt_tokens_details=SimpleNamespace(cached_tokens=usage.get("cached_tokens") or 0),
            ),
            model=model or "opencode-local",
        )

    def _stream_completion(self, base: str, mode: str, model: str | None, payload: dict[str, Any], timeout: float) -> Iterator[Any]:
        parse_delta = _STREAM_DELTA_PARSERS[mode]
        resp = self._post(base, _ENDPOINTS[mode], payload, timeout=timeout, stream=True)
        tool_call_states: dict[int, dict[str, Any]] = {}

        def _gen() -> Iterator[Any]:
            try:
                for event in _iter_sse_events(resp):
                    text, reasoning, tool_delta_raw, finish_reason = parse_delta(event)
                    tool_call_deltas = None
                    if tool_delta_raw:
                        tool_call_deltas = []
                        for raw in tool_delta_raw:
                            index = raw.get("index", 0)
                            state = tool_call_states.setdefault(index, {"id": None, "name": None})
                            if raw.get("id"):
                                state["id"] = raw["id"]
                            fn = raw.get("function") or {}
                            if fn.get("name"):
                                state["name"] = fn["name"]
                            tool_call_deltas.append(SimpleNamespace(
                                index=index, id=raw.get("id"), type="function",
                                function=SimpleNamespace(name=fn.get("name"), arguments=fn.get("arguments") or ""),
                            ))
                    if not (text or reasoning or tool_call_deltas or finish_reason):
                        continue
                    delta = SimpleNamespace(
                        role="assistant", content=text or None, tool_calls=tool_call_deltas,
                        reasoning_content=reasoning or None, reasoning=reasoning or None,
                    )
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                        model=model or "opencode-local", usage=None,
                    )
            finally:
                with contextlib.suppress(Exception):
                    resp.close()

        return _gen()
