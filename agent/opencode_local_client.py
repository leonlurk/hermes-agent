"""OpenAI-compatible shim that drives a local ``opencode serve`` subprocess over its HTTP
session API.

``opencode serve`` is NOT an OpenAI-compatible relay: ``/v1/chat/completions``, ``/v1/responses``
and ``/v1/messages`` do not exist on it — the server answers 200 with its SPA's HTML for any
unrecognized path, which silently defeats a naive health check or JSON parse. It is a session+SSE
agent server, the same one its own TUI talks to (verified against a live ``opencode serve``
v1.18.31 instance; see ``files/openapi-doc.json`` / ``files/NOTES.md`` for the captured evidence
this module is built from). A turn is: ``POST /session`` (model is set HERE, never on the prompt —
sending ``model`` on ``prompt_async`` reproducibly crashes the session with a SQLite
``NOT NULL constraint failed: session_message.seq`` error via ``session.next.agent.switched`` on
this server version) -> open ``GET /event`` (SSE) -> ``POST /session/{id}/prompt_async`` -> wait for
``session.idle`` / ``session.error`` -> ``GET /session/{id}/message`` as the ground truth of what the
turn produced (mirrors the TUI's own read-after-write pattern; SSE parts are not used to reconstruct
content because there is no confirmed evidence they are true deltas rather than full-state resends).

Like Copilot ACP, ``opencode serve`` is an AUTONOMOUS AGENT CLI with its OWN tools (bash/edit/read,
executed against ITS OWN cwd) — not a raw model backend Hermes' tool loop plugs into, and its wire
has no channel to hand it Hermes' custom tool schemas (``prompt_async.tools`` is only an
enable/disable map for opencode's OWN built-in tool ids, from ``GET /experimental/tool/ids``). So,
exactly like ``copilot_acp_client.py``: every one of opencode's built-in tools is disabled per
request, and Hermes' tool schemas travel IN as prompt text via the shared ``acp_openai_bridge``
helpers, parsed back OUT of the final text — same contract, HTTP+session/SSE transport instead of
stdio+ACP. Reasoning arrives as opencode's own ``reasoning`` message parts, kept on a distinct field
(``message.reasoning`` / ``reasoning_content``) rather than folded into the visible response text.

Unlike Copilot ACP (a short-lived session per request), ``opencode serve`` itself is meant to run as
a persistent local server, so the SUBPROCESS is a process-wide singleton reused across requests;
each Hermes turn still gets its own ephemeral opencode session (created, prompted, read, deleted),
which sidesteps opencode accumulating server-side conversation state Hermes does not control --
Hermes already resends the full transcript on every call, exactly like every other provider.
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

from agent.acp_openai_bridge import (
    completion_to_stream_chunks as _completion_to_stream_chunks,
    extract_tool_calls_from_text as _extract_tool_calls_from_text,
    render_tool_bridge_sections as _render_tool_bridge_sections,
)
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)

OPENCODE_LOCAL_MARKER_BASE_URL = "opencode-local://127.0.0.1"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_HEALTH_CHECK_TIMEOUT_SECONDS = 20.0
_HEALTH_CHECK_INTERVAL_SECONDS = 0.2
_EVENT_CONNECT_TIMEOUT_SECONDS = 10.0
# Fixed at both session create and prompt time: a mismatch (or a model on prompt_async) trips the
# `session.next.agent.switched` SQLite crash described in the module docstring.
_AGENT_PRESET = "build"

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool", "context": "Context"}
_PROMPT_PREAMBLE = (
    "You are being used as the active local-agent backend for Hermes.",
    "Do not use any of your own tools to take action — none are available to you here.",
    "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
    "If no tool is needed, answer normally.",
)


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        return content["content"].strip() if isinstance(content.get("content"), str) else json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts = [item if isinstance(item, str) else item["text"].strip() for item in content if isinstance(item, str)
                 or (isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip())]
        return "\n".join(parts).strip()
    return str(content).strip()


def _format_messages_as_prompt(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
) -> str:
    sections: list[str] = [*_PROMPT_PREAMBLE, *_render_tool_bridge_sections(tools, tool_choice)]
    transcript: list[str] = []
    for message in (m for m in messages if isinstance(m, dict)):
        role = str(message.get("role") or "unknown").strip().lower()
        if rendered := _render_message_content(message.get("content")):
            transcript.append(f"{_ROLE_LABELS.get(role, 'Context')}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _split_qualified_model(model: str | None) -> tuple[str, str]:
    """``"<providerID>/<modelID>"`` -> ``(providerID, modelID)``. ``modelID`` may itself contain
    ``/`` (e.g. OpenRouter's own ``qwen/qwen3.7-max``), so only the FIRST segment is the provider."""
    provider_id, _, model_id = str(model or "").strip().partition("/")
    return provider_id, model_id


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


def _get_json(base: str, path: str, timeout: float) -> Any:
    req = urllib.request.Request(base.rstrip("/") + path)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def _post_json(base: str, path: str, body: Any, timeout: float) -> tuple[int, Any]:
    data = json.dumps(body if body is not None else {}).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"opencode serve {path} failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:500]}") from exc
    return status, (json.loads(raw.decode("utf-8")) if raw else None)


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
        self._disabled_tools: dict[str, bool] | None = None
        self._disabled_tools_lock = threading.Lock()

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

    def disabled_tools_map(self, *, timeout_seconds: float) -> dict[str, bool]:
        """``{tool_id: False, ...}`` for every one of opencode's OWN built-in tools, so a turn
        never executes anything server-side — Hermes' tool loop stays the single source of truth
        for what actually runs. Fetched once per server and cached; a fetch failure degrades to an
        empty map (server default tool set stays enabled) rather than failing the whole turn."""
        with self._disabled_tools_lock:
            if self._disabled_tools is not None:
                return self._disabled_tools
        base = self.base_url(timeout_seconds=timeout_seconds)
        try:
            ids = _get_json(base, "/experimental/tool/ids", timeout_seconds) or []
            disabled = {str(i): False for i in ids if isinstance(i, str) and i.strip()}
        except Exception:
            logger.debug("opencode-local: could not fetch /experimental/tool/ids", exc_info=True)
            disabled = {}
        with self._disabled_tools_lock:
            self._disabled_tools = disabled
        return disabled

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
                with urllib.request.urlopen(base_url + "/global/health", timeout=1.0) as resp:
                    resp.read(1)
                healthy = True
                break
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    healthy = True  # server answered (even an error status) — it is up
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


# ── SSE plumbing ─────────────────────────────────────────────────────────────────────────────


def _iter_sse_json(resp: Any) -> Iterator[dict[str, Any]]:
    """Yield each ``data:`` line's JSON payload. opencode's ``/event`` stream carries no ``event:``
    field and one complete JSON object per ``data:`` line (no observed multi-line data blocks), so
    this stays a simple per-line parse rather than the buffer-until-blank-line SSE dance."""
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].lstrip(" ")
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except Exception:
            continue


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
        """Model ids as ``providerID/modelID`` (opencode's own model ids may themselves contain a
        ``/``, e.g. OpenRouter's ``qwen/qwen3.7-max``), from ``GET /config/providers`` — the local
        server's real catalog endpoint (there is no ``/v1/models``)."""
        base = self._server.base_url(timeout_seconds=timeout_seconds)
        data = _get_json(base, "/config/providers", timeout_seconds) or {}
        return [
            f"{provider['id']}/{model_id}"
            for provider in (data.get("providers") or [])
            if isinstance(provider, dict) and provider.get("id")
            for model_id in (provider.get("models") or {}).keys()
        ]

    # ── one turn ─────────────────────────────────────────────────────────────────────────────

    def _create_chat_completion(
        self, *, model: str | None = None, messages: list[dict[str, Any]] | None = None, timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None, stream: bool = False, **_: Any,
    ) -> Any:
        provider_id, model_id = _split_qualified_model(model)
        if not provider_id or not model_id:
            raise RuntimeError(
                f"opencode-local model ids must be qualified as 'providerID/modelID' (got {model!r}). "
                "Pick one from `hermes model` or GET /config/providers."
            )
        request_timeout = _effective_timeout(timeout)
        base = self._server.base_url(timeout_seconds=max(_HEALTH_CHECK_TIMEOUT_SECONDS, min(request_timeout, 120.0)))
        disabled_tools = self._server.disabled_tools_map(timeout_seconds=request_timeout)
        prompt_text = _format_messages_as_prompt(messages or [], tools=tools, tool_choice=tool_choice)
        response_text, reasoning = self._run_turn(
            base, provider_id, model_id, prompt_text, disabled_tools, timeout_seconds=request_timeout)
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        message = SimpleNamespace(
            content=cleaned_text, tool_calls=tool_calls, reasoning=reasoning or None, reasoning_content=reasoning or None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
            model=model or "opencode-local",
        )
        return _completion_to_stream_chunks(completion) if stream else completion

    def _run_turn(
        self, base: str, provider_id: str, model_id: str, prompt_text: str, disabled_tools: dict[str, bool],
        *, timeout_seconds: float,
    ) -> tuple[str, str]:
        session_body = {"title": "hermes-turn", "agent": _AGENT_PRESET, "model": {"providerID": provider_id, "id": model_id}}
        _, session = _post_json(base, "/session", session_body, timeout_seconds)
        sid = (session or {}).get("id")
        if not sid:
            raise RuntimeError("opencode serve did not return a session id.")
        try:
            connected = threading.Event()
            done = threading.Event()
            outcome: dict[str, Any] = {}
            watcher = threading.Thread(
                target=self._watch_events, args=(base, sid, connected, done, outcome, timeout_seconds), daemon=True)
            watcher.start()
            if not connected.wait(timeout=min(_EVENT_CONNECT_TIMEOUT_SECONDS, timeout_seconds)):
                raise TimeoutError("Timed out opening the opencode serve event stream.")

            prompt_body = {"agent": _AGENT_PRESET, "tools": disabled_tools, "parts": [{"type": "text", "text": prompt_text}]}
            status, _ = _post_json(base, f"/session/{sid}/prompt_async", prompt_body, timeout_seconds)
            if status not in (200, 204):
                raise RuntimeError(f"opencode serve rejected the prompt (HTTP {status}).")

            if not done.wait(timeout=timeout_seconds):
                with contextlib.suppress(Exception):
                    _post_json(base, f"/session/{sid}/abort", {}, 5.0)
                raise TimeoutError(f"opencode serve turn did not finish within {timeout_seconds:.0f}s.")
            if outcome.get("error"):
                raise RuntimeError(f"opencode serve turn failed: {outcome['error']}")
            return self._final_text_and_reasoning(base, sid, timeout_seconds)
        finally:
            with contextlib.suppress(Exception):
                req = urllib.request.Request(base.rstrip("/") + f"/session/{sid}", method="DELETE")
                urllib.request.urlopen(req, timeout=5.0).close()

    def _watch_events(
        self, base: str, sid: str, connected: threading.Event, done: threading.Event, outcome: dict[str, Any],
        timeout_seconds: float,
    ) -> None:
        """Consume the GLOBAL ``/event`` stream (opencode has no per-session subscribe filter),
        keeping only events for *sid*, until the turn finishes or fails."""
        req = urllib.request.Request(base.rstrip("/") + "/event")
        req.add_header("Accept", "text/event-stream")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                connected.set()
                for event in _iter_sse_json(resp):
                    if done.is_set():
                        return
                    props = event.get("properties") or {}
                    if props.get("sessionID") != sid:
                        continue
                    event_type = event.get("type")
                    if event_type == "session.idle":
                        done.set()
                        return
                    if event_type == "session.error":
                        error = props.get("error") or {}
                        outcome["error"] = (error.get("data") or {}).get("message") or error.get("name") or error
                        done.set()
                        return
                    if event_type == "session.status" and (props.get("status") or {}).get("type") == "idle":
                        done.set()
                        return
                    if event_type == "message.updated":
                        info = props.get("info") or {}
                        if info.get("role") == "assistant" and info.get("error"):
                            outcome["error"] = info["error"]
                            done.set()
                            return
        except Exception as exc:
            outcome.setdefault("error", f"{type(exc).__name__}: {exc}")
            connected.set()
            done.set()

    def _final_text_and_reasoning(self, base: str, sid: str, timeout_seconds: float) -> tuple[str, str]:
        """Ground truth of what the turn produced: the assistant message(s)' parts from
        ``GET /session/{id}/message``, not the SSE stream (see module docstring)."""
        data = _get_json(base, f"/session/{sid}/message", timeout_seconds) or []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for entry in data:
            if not isinstance(entry, dict) or (entry.get("info") or {}).get("role") != "assistant":
                continue
            for part in entry.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text" and not part.get("ignored"):
                    text_parts.append(part.get("text") or "")
                elif part_type == "reasoning":
                    reasoning_parts.append(part.get("text") or "")
        return "".join(text_parts), "".join(reasoning_parts)


def _effective_timeout(timeout: Any) -> float:
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    return max((float(v) for v in candidates if isinstance(v, (int, float))), default=_DEFAULT_TIMEOUT_SECONDS)
