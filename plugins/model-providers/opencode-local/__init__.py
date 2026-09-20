"""OpenCode Local provider profile.

``opencode-local`` does not speak OpenAI-over-HTTP against a hosted endpoint: it spawns
``opencode serve`` as a local subprocess and drives its session/SSE HTTP API (see
``agent/opencode_local_client.py`` for the protocol — verified against a live server, not the
hosted Zen/Go relay's wire shape), so the profile supplies its own client via
:meth:`ProviderProfile.create_client` — same seam as ``copilot-acp``, HTTP instead of stdio/ACP.
See ``hermes_cli/auth.py``'s unknown-provider hint for why a local server exists at all: OpenCode's
hosted Zen/Go relay 403s anonymous access from anything but its own client (FreeTierError), and the
local server IS that official client.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Model ids are "providerID/modelID" (opencode's own /config/providers catalog, not a flat name —
# a bare model id is ambiguous once opencode has more than one provider configured). This is
# opencode's OWN bundled "opencode" provider (OpenCode Zen's free tier, verified against a live
# server): the one catalog that exists on every install regardless of what other providers
# (OpenRouter, Anthropic, ...) a given user has separately configured inside their opencode CLI.
_FALLBACK_MODELS = (
    "opencode/big-pickle",
    "opencode/muse-spark-1.3-contributor-free",
    "opencode/muse-spark-1.2-contributor-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
    "opencode/mimo-v2.5-free",
    "opencode/ling-3.0-flash-fin-free",
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
    # OpencodeLocalClient always presents a chat_completions-shaped facade to Hermes (like
    # copilot-acp) regardless of which opencode-internal provider/model backs a given session.
    api_mode="chat_completions",
    env_vars=(),  # managed by the opencode CLI's own auth, not Hermes
    base_url="opencode-local://127.0.0.1",  # internal marker scheme, resolved to a real localhost port at spawn time
    auth_type="external_process",
    process_command="opencode",
    process_args=("serve",),
    process_command_env_vars=("HERMES_OPENCODE_LOCAL_COMMAND",),
    process_args_env_var="HERMES_OPENCODE_LOCAL_ARGS",
    fallback_models=_FALLBACK_MODELS,
    # No universal cheap aux pick: which models are fast/cheap depends entirely on what the
    # user's own opencode CLI has configured. Falls through to the main model (base default).
)

register_provider(opencode_local)
