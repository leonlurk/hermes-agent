"""Unit and integration tests for the OpenCode Local HTTP session-API shim.

Covers the things the plugin exists to get right, verified against evidence captured from a real
``opencode serve`` v1.18.31 instance (session-based, not OpenAI-wire — see the module docstring in
``agent/opencode_local_client.py``):
  1. model must be set at session-create, never at prompt_async (reproducible server crash otherwise)
  2. every one of opencode's own built-in tools is disabled per turn (Hermes' tool loop stays the
     only thing that actually executes anything)
  3. reasoning parts land on a structured field, never spliced into the visible response text
  4. tool calls the model emits via the text bridge are extracted into structured tool_calls, not
     left as visible prose
  5. the final result comes from GET /session/{id}/message (ground truth), not reconstructed from
     the SSE stream
  6. a session.error turn raises, a turn that never goes idle times out and is aborted
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from agent.opencode_local_client import (
    OpencodeLocalClient,
    _format_messages_as_prompt,
    _iter_sse_json,
    _render_message_content,
    _split_qualified_model,
)


# ── Small pure helpers ──────────────────────────────────────────────────────────────────────


class TestSplitQualifiedModel:
    def test_simple_provider_and_model(self):
        assert _split_qualified_model("opencode/big-pickle") == ("opencode", "big-pickle")

    def test_model_id_itself_contains_a_slash(self):
        # OpenRouter's own model ids are vendor/model — only the FIRST segment is the provider.
        assert _split_qualified_model("openrouter/qwen/qwen3.7-max") == ("openrouter", "qwen/qwen3.7-max")

    def test_unqualified_model_is_not_silently_accepted(self):
        provider_id, model_id = _split_qualified_model("big-pickle")
        assert provider_id == "big-pickle"
        assert model_id == ""  # caller must reject this, see TestUnqualifiedModelRejected

    def test_empty(self):
        assert _split_qualified_model(None) == ("", "")
        assert _split_qualified_model("") == ("", "")


class TestRenderMessageContent:
    def test_plain_string(self):
        assert _render_message_content("hello") == "hello"

    def test_none(self):
        assert _render_message_content(None) == ""

    def test_list_of_text_parts(self):
        content = [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]
        assert _render_message_content(content) == "line one\nline two"


class TestFormatMessagesAsPrompt:
    def test_includes_transcript_and_tool_bridge(self):
        messages = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "list files"}]
        tools = [{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {}}}]
        prompt = _format_messages_as_prompt(messages, tools=tools)
        assert "list files" in prompt
        assert "ls" in prompt
        assert "<tool_call>" in prompt  # the bridge contract instructs the model to emit these

    def test_instructs_the_model_not_to_use_its_own_tools(self):
        prompt = _format_messages_as_prompt([{"role": "user", "content": "hi"}])
        assert "own tools" in prompt.lower()


# ── SSE plumbing ────────────────────────────────────────────────────────────────────────────


class TestIterSseJson:
    def test_yields_one_event_per_data_line(self):
        lines = [b'data: {"a": 1}\n', b"\n", b'data: {"a": 2}\n', b"\n"]
        assert list(_iter_sse_json(lines)) == [{"a": 1}, {"a": 2}]

    def test_comment_lines_are_ignored(self):
        lines = [b": keep-alive\n", b'data: {"a": 1}\n']
        assert list(_iter_sse_json(lines)) == [{"a": 1}]

    def test_malformed_event_is_skipped_not_fatal(self):
        lines = [b"data: not json\n", b'data: {"a": 1}\n']
        assert list(_iter_sse_json(lines)) == [{"a": 1}]

    def test_streams_incrementally(self):
        def _lines():
            yield b'data: {"a": 1}\n'
            yield b'data: {"a": 2}\n'
            raise AssertionError("must not need to read past the second event to get it")

        gen = _iter_sse_json(_lines())
        assert next(gen) == {"a": 1}
        assert next(gen) == {"a": 2}


# ── End-to-end against a stub opencode serve ───────────────────────────────────────────────


class _StubOpencodeServer:
    """A minimal stand-in for ``opencode serve``'s session+SSE API. ``on_prompt_events`` are
    broadcast on the global ``/event`` stream (sessionID auto-filled) the instant a
    ``prompt_async`` POST lands, mirroring how the real server pushes events asynchronously."""

    def __init__(self, *, on_prompt_events=None, final_message_parts=None, tool_ids=("bash", "edit", "read"),
                 config_providers=None, prompt_status=204):
        self.requests: list[tuple[str, str, dict]] = []
        self.sessions: dict[str, dict] = {}
        self._on_prompt_events = list(on_prompt_events or [])
        self._final_message_parts = final_message_parts if final_message_parts is not None else []
        self._tool_ids = list(tool_ids)
        self._config_providers = config_providers or {"providers": [], "default": {}}
        self._prompt_status = prompt_status
        self._sse_clients: list[queue.Queue] = []
        self._sid_counter = 0

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _json(self, obj, status=200):
                payload = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/global/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                if self.path == "/event":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.flush()
                    q: queue.Queue = queue.Queue()
                    outer._sse_clients.append(q)
                    try:
                        while True:
                            item = q.get()
                            if item is None:
                                break
                            self.wfile.write(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        with_ = outer._sse_clients
                        if q in with_:
                            with_.remove(q)
                    return
                if self.path == "/experimental/tool/ids":
                    self._json(outer._tool_ids)
                    return
                if self.path == "/config/providers":
                    self._json(outer._config_providers)
                    return
                if self.path.startswith("/session/") and self.path.endswith("/message"):
                    self._json([{"info": {"role": "assistant"}, "parts": outer._final_message_parts}])
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(("POST", self.path, body))
                if self.path == "/session":
                    outer._sid_counter += 1
                    sid = f"ses_test{outer._sid_counter}"
                    outer.sessions[sid] = body
                    self._json({"id": sid, **body})
                    return
                if self.path.endswith("/prompt_async"):
                    sid = self.path.split("/")[2]
                    for event in outer._on_prompt_events:
                        event = dict(event)
                        event.setdefault("properties", {})
                        event["properties"] = {"sessionID": sid, **event["properties"]}
                        for client_queue in list(outer._sse_clients):
                            client_queue.put(event)
                    if outer._prompt_status == 204:
                        self.send_response(204)
                        self.end_headers()
                    else:
                        self._json({}, status=outer._prompt_status)
                    return
                if self.path.endswith("/abort"):
                    self._json({})
                    return
                self.send_response(404)
                self.end_headers()

            def do_DELETE(self):
                outer.requests.append(("DELETE", self.path, {}))
                self.send_response(200)
                self.end_headers()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self):
        for q in list(self._sse_clients):
            q.put(None)
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _reset_server_singleton(monkeypatch):
    # _OpencodeLocalServer is a process-wide singleton keyed on (command, args); reset it per test
    # so one test's (now torn-down) stub server URL and cached tool-ids never leak into another.
    from agent.opencode_local_client import _OpencodeLocalServer

    monkeypatch.setattr(_OpencodeLocalServer, "_instances", {})


@pytest.fixture
def stub_server():
    server = _StubOpencodeServer()
    try:
        yield server
    finally:
        server.shutdown()


def _client_for(stub, monkeypatch) -> OpencodeLocalClient:
    monkeypatch.setenv("HERMES_OPENCODE_LOCAL_URL", stub.base_url)
    return OpencodeLocalClient(command="opencode", args=["serve"])


class TestUnqualifiedModelRejected:
    def test_bare_model_id_raises_before_any_request(self, stub_server, monkeypatch):
        client = _client_for(stub_server, monkeypatch)
        with pytest.raises(RuntimeError, match="providerID/modelID"):
            client.chat.completions.create(model="big-pickle", messages=[{"role": "user", "content": "hi"}])
        assert stub_server.requests == []  # rejected locally — never reached the server


class TestOpencodeLocalTurnHappyPath:
    def test_model_set_at_create_never_at_prompt(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": "DONE"}]
        client = _client_for(stub_server, monkeypatch)
        client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])

        create_calls = [r for r in stub_server.requests if r[1] == "/session"]
        prompt_calls = [r for r in stub_server.requests if r[1].endswith("/prompt_async")]
        assert len(create_calls) == 1
        assert create_calls[0][2]["model"] == {"providerID": "opencode", "id": "big-pickle"}
        assert "model" not in prompt_calls[0][2]  # the reproducible SQLite crash trigger

    def test_disables_every_builtin_tool_on_the_prompt(self, stub_server, monkeypatch):
        stub_server._tool_ids = ["bash", "edit", "read"]
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": "DONE"}]
        client = _client_for(stub_server, monkeypatch)
        client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])

        prompt_call = next(r for r in stub_server.requests if r[1].endswith("/prompt_async"))
        assert prompt_call[2]["tools"] == {"bash": False, "edit": False, "read": False}

    def test_reasoning_lands_on_a_separate_field(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [
            {"type": "reasoning", "text": "thinking about it..."},
            {"type": "text", "text": "The answer is 4."},
        ]
        client = _client_for(stub_server, monkeypatch)
        result = client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "2+2?"}])

        assert result.choices[0].message.content == "The answer is 4."
        assert result.choices[0].message.reasoning == "thinking about it..."
        assert "thinking" not in result.choices[0].message.content

    def test_tool_call_bridge_is_extracted_not_left_as_prose(self, stub_server, monkeypatch):
        tool_call_text = (
            "<tool_call>"
            '{"id":"call_1","type":"function","function":{"name":"ls","arguments":"{}"}}'
            "</tool_call>"
        )
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": tool_call_text}]
        client = _client_for(stub_server, monkeypatch)
        result = client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "list files"}])

        message = result.choices[0].message
        assert message.content in (None, "")
        assert result.choices[0].finish_reason == "tool_calls"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].function.name == "ls"

    def test_uses_final_message_endpoint_not_sse_for_content(self, stub_server, monkeypatch):
        # The SSE stream carries no text/reasoning here at all — only the completion signal — yet
        # the final content still comes through correctly from GET /session/{id}/message.
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": "ground truth wins"}]
        client = _client_for(stub_server, monkeypatch)
        result = client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])
        assert result.choices[0].message.content == "ground truth wins"

    def test_session_deleted_after_the_turn(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": "DONE"}]
        client = _client_for(stub_server, monkeypatch)
        client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])
        deletes = [r for r in stub_server.requests if r[0] == "DELETE"]
        assert len(deletes) == 1


class TestOpencodeLocalTurnFailureModes:
    def test_session_error_event_raises(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = [
            {"type": "session.error", "properties": {"error": {"data": {"message": "boom"}}}},
        ]
        client = _client_for(stub_server, monkeypatch)
        with pytest.raises(RuntimeError, match="boom"):
            client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])

    def test_turn_that_never_idles_times_out_and_aborts(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = []  # never signals completion
        client = _client_for(stub_server, monkeypatch)
        # Generous budget: this must give the SSE connect + prompt POST room to complete under
        # test-runner scheduling noise, so the timeout is the DONE wait (which aborts), not a
        # spurious connect-stage timeout (which correctly would not abort).
        with pytest.raises(TimeoutError):
            client.chat.completions.create(
                model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}], timeout=3.0)
        aborts = [r for r in stub_server.requests if r[1].endswith("/abort")]
        assert len(aborts) == 1


class TestOpencodeLocalStreaming:
    def test_stream_preserves_embedded_newlines(self, stub_server, monkeypatch):
        stub_server._on_prompt_events = [{"type": "session.idle", "properties": {}}]
        stub_server._final_message_parts = [{"type": "text", "text": "line one\nline two"}]
        client = _client_for(stub_server, monkeypatch)
        stream = client.chat.completions.create(
            model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}], stream=True)
        chunks = list(stream)
        contents = [c.choices[0].delta.content for c in chunks if getattr(c, "choices", None) and c.choices[0].delta.content]
        assert contents == ["line one\nline two"]


class TestOpencodeLocalListModels:
    def test_flattens_provider_and_model_ids(self, stub_server, monkeypatch):
        stub_server._config_providers = {
            "providers": [
                {"id": "opencode", "models": {"big-pickle": {}, "mimo-v2.5-free": {}}},
                {"id": "openrouter", "models": {"qwen/qwen3.7-max": {}}},
            ],
            "default": {},
        }
        client = _client_for(stub_server, monkeypatch)
        models = client.list_models()
        assert set(models) == {"opencode/big-pickle", "opencode/mimo-v2.5-free", "openrouter/qwen/qwen3.7-max"}


class TestOpencodeLocalServerLifecycle:
    def test_reuses_external_server_without_spawning(self, stub_server, monkeypatch):
        monkeypatch.setenv("HERMES_OPENCODE_LOCAL_URL", stub_server.base_url)
        with patch("subprocess.Popen") as mock_popen:
            client = OpencodeLocalClient(command="opencode", args=["serve"])
            client.list_models()
            mock_popen.assert_not_called()
            client.close()

    def test_missing_binary_raises_actionable_error(self, monkeypatch):
        monkeypatch.delenv("HERMES_OPENCODE_LOCAL_URL", raising=False)
        client = OpencodeLocalClient(command="definitely-not-a-real-opencode-binary-xyz", args=["serve"])
        with pytest.raises(RuntimeError, match="Could not start opencode-local command"):
            client.chat.completions.create(model="opencode/big-pickle", messages=[{"role": "user", "content": "hi"}])
