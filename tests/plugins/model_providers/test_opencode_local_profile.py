"""Unit tests for the OpenCode Local provider profile.

``opencode-local`` spawns ``opencode serve`` as a subprocess and drives its local session/SSE
HTTP API, mirroring the ``copilot-acp`` plugin's ``auth_type="external_process"`` pattern
(stdio/ACP swapped for HTTP). See ``agent/opencode_local_client.py`` for the subprocess lifecycle
and turn protocol; this file only covers the ``ProviderProfile`` contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def opencode_local_profile():
    import model_tools  # noqa: F401 — triggers discovery
    import providers

    profile = providers.get_provider_profile("opencode-local")
    assert profile is not None, "opencode-local provider profile must be registered"
    return profile


class TestOpencodeLocalProfileIdentity:
    def test_name(self, opencode_local_profile):
        assert opencode_local_profile.name == "opencode-local"

    def test_aliases(self, opencode_local_profile):
        assert "opencode-serve" in opencode_local_profile.aliases

    def test_auth_type(self, opencode_local_profile):
        assert opencode_local_profile.auth_type == "external_process"

    def test_api_mode_is_the_bridge_facade(self, opencode_local_profile):
        # OpencodeLocalClient always presents a chat_completions-shaped facade to Hermes, like
        # copilot-acp — regardless of which opencode-internal provider/model backs the session.
        assert opencode_local_profile.api_mode == "chat_completions"

    def test_no_env_vars_subprocess_owns_auth(self, opencode_local_profile):
        assert opencode_local_profile.env_vars == ()

    def test_process_command(self, opencode_local_profile):
        assert opencode_local_profile.process_command == "opencode"
        assert opencode_local_profile.process_args == ("serve",)

    def test_process_env_overrides(self, opencode_local_profile):
        assert "HERMES_OPENCODE_LOCAL_COMMAND" in opencode_local_profile.process_command_env_vars
        assert opencode_local_profile.process_args_env_var == "HERMES_OPENCODE_LOCAL_ARGS"

    def test_fallback_models_are_qualified_provider_model_ids(self, opencode_local_profile):
        # A bare model id is ambiguous once more than one provider is configured inside opencode
        # (see OpencodeLocalClient._split_qualified_model) — every fallback must carry a provider.
        for model in opencode_local_profile.fallback_models:
            provider_id, _, model_id = model.partition("/")
            assert provider_id and model_id, f"not a qualified providerID/modelID: {model!r}"

    def test_display_name_and_description(self, opencode_local_profile):
        assert "OpenCode" in opencode_local_profile.display_name
        assert opencode_local_profile.description


class TestOpencodeLocalRegistryIntegrity:
    def test_registered_and_discoverable(self):
        import model_tools  # noqa: F401
        import providers

        profile = providers.get_provider_profile("opencode-local")
        assert profile is not None

    def test_alias_lookup(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("opencode-serve") is not None
        assert providers.get_provider_profile("opencode-local-server") is not None

    def test_unknown_returns_none(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("opencode-local-nonexistent") is None

    def test_resolve_provider_full(self):
        from hermes_cli.providers import resolve_provider_full

        resolved = resolve_provider_full("opencode-local", {}, [])
        assert resolved is not None and resolved.id == "opencode-local"


class TestOpencodeLocalCreateClient:
    def test_create_client_returns_local_http_shim(self, opencode_local_profile):
        from agent.opencode_local_client import OpencodeLocalClient

        client = opencode_local_profile.create_client(api_key="opencode-local", base_url="opencode-local://127.0.0.1")
        assert isinstance(client, OpencodeLocalClient)
        # Consumed the same way the standard openai.OpenAI client is.
        assert hasattr(client.chat.completions, "create")

    def test_client_is_not_re_wrapped_by_transport_or_async_adapters(self, opencode_local_profile):
        from agent.opencode_local_client import OpencodeLocalClient

        assert OpencodeLocalClient.HERMES_SKIP_TRANSPORT_WRAP is True
        assert OpencodeLocalClient.HERMES_SKIP_ASYNC_WRAP is True


class TestOpencodeLocalFetchModels:
    def test_missing_cli_returns_none(self, opencode_local_profile):
        from hermes_cli.auth_constants import AuthError

        with patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            side_effect=AuthError("no opencode binary", code="missing_external_process_cli"),
        ):
            assert opencode_local_profile.fetch_models() is None

    def test_spawn_failure_returns_none(self, opencode_local_profile):
        creds = {"provider": "opencode-local", "api_key": "opencode-local", "base_url": "opencode-local://127.0.0.1",
                  "command": "opencode", "args": ["serve"], "source": "process"}
        with patch("hermes_cli.auth.resolve_external_process_provider_credentials", return_value=creds), \
             patch("agent.opencode_local_client.OpencodeLocalClient.list_models", side_effect=RuntimeError("boom")):
            assert opencode_local_profile.fetch_models() is None

    def test_successful_probe_returns_model_list(self, opencode_local_profile):
        creds = {"provider": "opencode-local", "api_key": "opencode-local", "base_url": "opencode-local://127.0.0.1",
                  "command": "opencode", "args": ["serve"], "source": "process"}
        with patch("hermes_cli.auth.resolve_external_process_provider_credentials", return_value=creds), \
             patch("agent.opencode_local_client.OpencodeLocalClient.list_models", return_value=["claude-sonnet-4-6", "gpt-5.1-codex"]):
            assert opencode_local_profile.fetch_models() == ["claude-sonnet-4-6", "gpt-5.1-codex"]

    def test_empty_probe_result_returns_none(self, opencode_local_profile):
        creds = {"provider": "opencode-local", "api_key": "opencode-local", "base_url": "opencode-local://127.0.0.1",
                  "command": "opencode", "args": ["serve"], "source": "process"}
        with patch("hermes_cli.auth.resolve_external_process_provider_credentials", return_value=creds), \
             patch("agent.opencode_local_client.OpencodeLocalClient.list_models", return_value=[]):
            assert opencode_local_profile.fetch_models() is None
