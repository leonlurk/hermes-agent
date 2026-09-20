"""OpenCode Local provider profile.

``opencode-local`` does not speak OpenAI-over-HTTP against a hosted endpoint: it spawns
``opencode serve`` as a local subprocess and drives its HTTP API, so the profile supplies its own
client via :meth:`ProviderProfile.create_client` — same seam as ``copilot-acp``, HTTP instead of
stdio/ACP. See ``hermes_cli/auth.py``'s unknown-provider hint for why a local server exists at
all: OpenCode's hosted Zen/Go relay 403s anonymous access from anything but its own client
(FreeTierError), and the local server IS that official client.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Curated fallback catalog spanning every wire dialect opencode-local routes between (see
# ``agent.opencode_local_client.opencode_local_model_api_mode``): Muse Spark / GPT / Grok ->
# codex_responses, Claude / MiniMax / Qwen -> anthropic_messages, everything else -> chat_completions.
_FALLBACK_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gpt-5.1-codex",
    "grok-4.1",
    "muse-spark-1.3-contributor",
    "minimax-m2.5",
    "qwen3.7-max",
    "glm-5.2",
    "deepseek-v4-pro",
)


class OpencodeLocalProfile(ProviderProfile):
    """OpenCode Local — spawns ``opencode serve``, no hosted relay, no API key."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the local-HTTP shim rather than an ``openai.OpenAI`` client."""
        from agent.opencode_local_client import OpencodeLocalClient

        return OpencodeLocalClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 15.0
    ) -> list[str] | None:
        """Live catalog from the local server, spawning it if needed. ``api_key`` / ``base_url``
        are ignored: the subprocess owns its own auth and picks its own local port. ``None`` when
        the CLI is missing, the spawn fails, or the probe times out — callers fall back to
        ``fallback_models``."""
        from hermes_cli.auth import resolve_external_process_provider_credentials

        try:
            creds = resolve_external_process_provider_credentials(self.name)
            client = self.create_client(
                api_key=creds.get("api_key"), base_url=creds.get("base_url"),
                command=creds.get("command"), args=creds.get("args"))
            return client.list_models(timeout_seconds=timeout) or None
        except Exception:
            # Missing CLI (AuthError), failed/timed-out spawn (RuntimeError / TimeoutError), or a
            # catalog probe failure — the base fetch_models contract is "None if the fetch failed".
            return None


opencode_local = OpencodeLocalProfile(
    name="opencode-local", aliases=("opencode-serve", "opencode-local-server"),
    display_name="OpenCode Local", description="OpenCode CLI (local `opencode serve` subprocess, no relay)",
    signup_url="https://opencode.ai",
    api_mode="chat_completions",  # hint/fallback only — real routing is per-model, inside the client
    env_vars=(),  # managed by the opencode CLI's own auth, not Hermes
    base_url="opencode-local://127.0.0.1",  # internal marker scheme, resolved to a real localhost port at spawn time
    auth_type="external_process",
    process_command="opencode",
    process_args=("serve",),
    process_command_env_vars=("HERMES_OPENCODE_LOCAL_COMMAND",),
    process_args_env_var="HERMES_OPENCODE_LOCAL_ARGS",
    fallback_models=_FALLBACK_MODELS,
    default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(opencode_local)
