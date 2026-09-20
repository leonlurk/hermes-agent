"""Unit and integration tests for the OpenCode Local HTTP shim.

Covers the four things the plugin exists to get right:
  1. per-model wire-dialect routing (mirrors ``_OPENCODE_API_MODE_PREFIXES``)
  2. reasoning/thinking arrives as a structured field, never spliced into ``content``
  3. tool calls arrive as structured ``tool_calls``, never as prose the model has to be coaxed
     into emitting
  4. streaming forwards each SSE chunk as it arrives, unbuffered
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from agent.opencode_local_client import (
    OpencodeLocalClient,
    _iter_sse_events,
    _parse_anthropic_response,
    _parse_chat_response,
    _parse_responses_response,
    _to_anthropic_payload,
    _to_chat_payload,
    _to_responses_payload,
    opencode_local_model_api_mode,
)


# ── Per-model api_mode routing ──────────────────────────────────────────────────────────────


class TestOpencodeLocalModelApiMode:
    @pytest.mark.parametrize("model", [
        "muse-spark-1.3-contributor", "muse-spark-1.2", "gpt-5.1-codex", "gpt-5.1", "grok-4.1", "grok-4.1-fast",
    ])
    def test_codex_responses_family(self, model):
        assert opencode_local_model_api_mode(model) == "codex_responses"

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6", "claude-opus-4-6", "minimax-m2.5", "minimax-m3", "qwen3.7-max", "qwen3-max",
    ])
    def test_anthropic_messages_family(self, model):
        assert opencode_local_model_api_mode(model) == "anthropic_messages"

    @pytest.mark.parametrize("model", ["glm-5.2", "deepseek-v4-pro", "kimi-k2.7", "mimo-v2.5", "", None])
    def test_chat_completions_fallback(self, model):
        assert opencode_local_model_api_mode(model) == "chat_completions"

    def test_case_insensitive(self):
        assert opencode_local_model_api_mode("Claude-Sonnet-4-6") == "anthropic_messages"
        assert opencode_local_model_api_mode("GPT-5.1-Codex") == "codex_responses"

    def test_a_single_fixed_mode_is_not_used_for_every_model(self):
        modes = {opencode_local_model_api_mode(m) for m in
                 ("claude-sonnet-4-6", "gpt-5.1-codex", "glm-5.2")}
        assert modes == {"anthropic_messages", "codex_responses", "chat_completions"}


# ── Payload builders ────────────────────────────────────────────────────────────────────────


class TestPayloadBuilders:
    _MESSAGES = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
    ]
    _TOOLS = [{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {}}}]

    def test_chat_payload_passes_messages_through(self):
        payload = _to_chat_payload("glm-5.2", self._MESSAGES, self._TOOLS, "auto", False)
        assert payload["model"] == "glm-5.2"
        assert payload["messages"] == self._MESSAGES
        assert payload["tools"] == self._TOOLS
        assert payload["tool_choice"] == "auto"
        assert payload["stream"] is False

    def test_responses_payload_extracts_system_as_instructions(self):
        payload = _to_responses_payload("gpt-5.1-codex", self._MESSAGES, self._TOOLS, None, True)
        assert payload["instructions"] == "You are terse."
        assert payload["stream"] is True
        types = [item["type"] for item in payload["input"]]
        assert "function_call" in types
        assert "function_call_output" in types
        call = next(i for i in payload["input"] if i["type"] == "function_call")
        assert call["name"] == "ls" and call["call_id"] == "call_1"
        output = next(i for i in payload["input"] if i["type"] == "function_call_output")
        assert output["call_id"] == "call_1" and output["output"] == "README.md"

    def test_anthropic_payload_moves_system_out_of_messages(self):
        payload = _to_anthropic_payload("claude-sonnet-4-6", self._MESSAGES, self._TOOLS, "auto", False, max_tokens=4096)
        assert payload["system"] == "You are terse."
        assert all(m["role"] != "system" for m in payload["messages"])
        assert payload["max_tokens"] == 4096
        assert payload["tools"][0]["name"] == "ls"
        assert payload["tool_choice"] == {"type": "auto"}

    def test_anthropic_payload_converts_tool_call_and_result(self):
        payload = _to_anthropic_payload("claude-sonnet-4-6", self._MESSAGES, self._TOOLS, None, False, max_tokens=4096)
        assistant_msg = next(m for m in payload["messages"] if m["role"] == "assistant")
        tool_use = next(b for b in assistant_msg["content"] if b["type"] == "tool_use")
        assert tool_use["name"] == "ls" and tool_use["id"] == "call_1"
        tool_result_msg = payload["messages"][-1]
        result_block = tool_result_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "call_1"
        assert result_block["content"] == "README.md"

    def test_anthropic_tool_choice_none_drops_tools(self):
        payload = _to_anthropic_payload("claude-sonnet-4-6", self._MESSAGES, self._TOOLS, "none", False, max_tokens=4096)
        assert "tools" not in payload


# ── Response parsing: reasoning and tool calls must be structured, not prose ──────────────────


class TestResponseParsing:
    def test_chat_response_reasoning_is_not_appended_to_content(self):
        data = {"choices": [{"message": {
            "content": "The answer is 4.", "reasoning_content": "2 + 2 = 4, so the answer is 4.",
        }, "finish_reason": "stop"}]}
        content, reasoning, tool_calls, finish = _parse_chat_response(data)
        assert content == "The answer is 4."
        assert reasoning == "2 + 2 = 4, so the answer is 4."
        assert tool_calls == []
        assert finish == "stop"

    def test_chat_response_tool_calls_are_structured(self):
        data = {"choices": [{"message": {
            "content": None, "tool_calls": [
                {"id": "call_9", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
        }, "finish_reason": "tool_calls"}]}
        content, reasoning, tool_calls, finish = _parse_chat_response(data)
        assert content == ""
        assert finish == "tool_calls"
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "read_file"
        assert json.loads(tool_calls[0].function.arguments) == {"path": "a.py"}

    def test_responses_response_splits_reasoning_text_and_function_calls(self):
        data = {"output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking about it..."}]},
            {"type": "message", "content": [{"type": "output_text", "text": "done."}]},
            {"type": "function_call", "call_id": "call_5", "name": "write_file", "arguments": '{"path": "b.py"}'},
        ]}
        content, reasoning, tool_calls, finish = _parse_responses_response(data)
        assert content == "done."
        assert reasoning == "thinking about it..."
        assert finish == "tool_calls"
        assert tool_calls[0].function.name == "write_file"

    def test_anthropic_response_thinking_block_becomes_reasoning(self):
        data = {"content": [
            {"type": "thinking", "thinking": "Let me reason about this."},
            {"type": "text", "text": "Here is the result."},
        ], "stop_reason": "end_turn"}
        content, reasoning, tool_calls, finish = _parse_anthropic_response(data)
        assert content == "Here is the result."
        assert reasoning == "Let me reason about this."
        assert tool_calls == []
        assert finish == "stop"

    def test_anthropic_response_tool_use_block_becomes_tool_call(self):
        data = {"content": [
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
        ], "stop_reason": "tool_use"}
        content, reasoning, tool_calls, finish = _parse_anthropic_response(data)
        assert finish == "tool_calls"
        assert tool_calls[0].function.name == "bash"
        assert json.loads(tool_calls[0].function.arguments) == {"command": "ls"}


# ── SSE plumbing ────────────────────────────────────────────────────────────────────────────


class TestIterSseEvents:
    def test_yields_one_event_per_data_block(self):
        lines = [b'data: {"a": 1}\n', b"\n", b'data: {"a": 2}\n', b"\n", b"data: [DONE]\n", b"\n"]
        events = list(_iter_sse_events(lines))
        assert events == [{"a": 1}, {"a": 2}]

    def test_multiline_data_block_is_joined(self):
        lines = [b'data: {"a":\n', b'data: 1}\n', b"\n"]
        events = list(_iter_sse_events(lines))
        assert events == [{"a": 1}]

    def test_comment_lines_are_ignored(self):
        lines = [b": keep-alive\n", b'data: {"a": 1}\n', b"\n"]
        events = list(_iter_sse_events(lines))
        assert events == [{"a": 1}]

    def test_malformed_event_is_skipped_not_fatal(self):
        lines = [b"data: not json\n", b"\n", b'data: {"a": 1}\n', b"\n"]
        events = list(_iter_sse_events(lines))
        assert events == [{"a": 1}]

    def test_streams_incrementally_not_batched(self):
        """A generator that reads three chunks one at a time must not need the fourth
        (terminating) line before yielding the first two events."""
        seen: list[dict] = []

        def _lines():
            yield b'data: {"a": 1}\n'
            yield b"\n"
            seen.append("first event should already have been consumable")
            yield b'data: {"a": 2}\n'
            yield b"\n"
            raise AssertionError("must not need to read past the second event to get it")

        gen = _iter_sse_events(_lines())
        assert next(gen) == {"a": 1}
        assert next(gen) == {"a": 2}


# ── End-to-end against a stub local server ─────────────────────────────────────────────────


class _StubOpencodeServer:
    """A minimal HTTP stand-in for ``opencode serve`` that records the request path/body and
    replies with a canned response per endpoint, streaming or not."""

    def __init__(self):
        self.requests: list[tuple[str, dict]] = []
        handler = self._make_handler()
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, body))
                if self.path == "/v1/chat/completions":
                    self._chat(body)
                elif self.path == "/v1/responses":
                    self._responses(body)
                elif self.path == "/v1/messages":
                    self._messages(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _chat(self, body):
                if body.get("stream"):
                    self._sse([
                        {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking "}, "finish_reason": None}]},
                        {"choices": [{"index": 0, "delta": {"reasoning_content": "more\nlines"}, "finish_reason": None}]},
                        {"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]},
                    ])
                    return
                self._json({"choices": [{"message": {
                    "content": "answer", "reasoning_content": "because",
                }, "finish_reason": "stop"}], "usage": {}})

            def _responses(self, body):
                self._json({"output": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "codex thinking"}]},
                    {"type": "message", "content": [{"type": "output_text", "text": "codex answer"}]},
                ]})

            def _messages(self, body):
                self._json({"content": [
                    {"type": "thinking", "thinking": "claude thinking"},
                    {"type": "text", "text": "claude answer"},
                ], "stop_reason": "end_turn"})

            def _json(self, obj):
                payload = json.dumps(obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def _sse(self, events):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        return Handler

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def stub_server():
    server = _StubOpencodeServer()
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture(autouse=True)
def _reset_server_singleton(monkeypatch):
    # ``_OpencodeLocalServer`` is a process-wide singleton keyed on (command, args) so a real
    # `opencode serve` is reused across requests within one process — but that means it would
    # also cache one test's (now torn-down) stub server's URL for every later test that builds a
    # client with the same command/args. Reset it so each test gets its own resolution.
    from agent.opencode_local_client import _OpencodeLocalServer

    monkeypatch.setattr(_OpencodeLocalServer, "_instances", {})


@pytest.fixture
def local_client(stub_server, monkeypatch):
    monkeypatch.setenv("HERMES_OPENCODE_LOCAL_URL", stub_server.base_url)
    client = OpencodeLocalClient(command="opencode", args=["serve"])
    yield client
    client.close()


class TestOpencodeLocalClientRouting:
    def test_chat_family_model_hits_chat_completions_endpoint(self, local_client, stub_server):
        result = local_client.chat.completions.create(model="glm-5.2", messages=[{"role": "user", "content": "hi"}])
        assert stub_server.requests[0][0] == "/v1/chat/completions"
        assert result.choices[0].message.content == "answer"
        # Reasoning is a distinct field, never folded into the visible response text.
        assert result.choices[0].message.reasoning == "because"
        assert "because" not in result.choices[0].message.content

    def test_codex_family_model_hits_responses_endpoint(self, local_client, stub_server):
        result = local_client.chat.completions.create(model="gpt-5.1-codex", messages=[{"role": "user", "content": "hi"}])
        assert stub_server.requests[0][0] == "/v1/responses"
        assert result.choices[0].message.content == "codex answer"
        assert result.choices[0].message.reasoning == "codex thinking"

    def test_claude_family_model_hits_messages_endpoint(self, local_client, stub_server):
        result = local_client.chat.completions.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
        assert stub_server.requests[0][0] == "/v1/messages"
        assert result.choices[0].message.content == "claude answer"
        assert result.choices[0].message.reasoning == "claude thinking"


class TestOpencodeLocalClientStreaming:
    def test_stream_forwards_each_chunk_preserving_line_breaks(self, local_client, stub_server):
        stream = local_client.chat.completions.create(
            model="glm-5.2", messages=[{"role": "user", "content": "hi"}], stream=True,
        )
        chunks = list(stream)
        reasoning_chunks = [c.choices[0].delta.reasoning for c in chunks if c.choices[0].delta.reasoning]
        assert reasoning_chunks == ["thinking ", "more\nlines"]
        # The embedded newline from the server survives untouched — no re-flowing/re-chunking.
        assert "\n" in reasoning_chunks[1]
        content_chunks = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        assert content_chunks == ["answer"]
        assert chunks[-1].choices[0].finish_reason == "stop"


class TestOpencodeLocalServerLifecycle:
    def test_reuses_external_server_without_spawning(self, stub_server, monkeypatch):
        """HERMES_OPENCODE_LOCAL_URL lets an operator-managed `opencode serve` be reused —
        no subprocess spawned, so nothing for this process to orphan."""
        monkeypatch.setenv("HERMES_OPENCODE_LOCAL_URL", stub_server.base_url)
        with patch("subprocess.Popen") as mock_popen:
            client = OpencodeLocalClient(command="opencode", args=["serve"])
            client.chat.completions.create(model="glm-5.2", messages=[{"role": "user", "content": "hi"}])
            mock_popen.assert_not_called()
            client.close()

    def test_missing_binary_raises_actionable_error(self, monkeypatch):
        monkeypatch.delenv("HERMES_OPENCODE_LOCAL_URL", raising=False)
        client = OpencodeLocalClient(command="definitely-not-a-real-opencode-binary-xyz", args=["serve"])
        with pytest.raises(RuntimeError, match="Could not start opencode-local command"):
            client.chat.completions.create(model="glm-5.2", messages=[{"role": "user", "content": "hi"}])
