"""Configuration: environment settings plus the on-disk gateway config.

Two separate things live here, deliberately:

* ``Settings``      — deployment knobs (database URL, ports, timeouts). From env.
* ``GatewayConfig`` — the *routing* configuration: which providers exist, what the
                      aliases ``fast``/``smart``/``auto`` resolve to, retry policy.
                      From ``data/gateway_config.sample.json``.

Keeping them apart means you can point the same routing config at a different
database, or swap mock providers for real ones, without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    """Deployment settings, overridable by environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="PRISM_", env_file=".env", extra="ignore")

    # --- storage -------------------------------------------------------------
    # SQLite by default so the gateway runs with zero setup. Point this at
    # postgresql+asyncpg://... to use Postgres (pgvector is used automatically
    # when the extension is present). See docs/ARCHITECTURE.md.
    database_url: str = f"sqlite+aiosqlite:///{REPO_ROOT / 'prism.db'}"

    # --- files ---------------------------------------------------------------
    gateway_config_path: Path = DATA_DIR / "gateway_config.sample.json"
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    model_pricing_path: Path = DATA_DIR / "model_pricing.json"
    seed_keys_path: Path = DATA_DIR / "seed_keys.json"

    # --- upstream behaviour --------------------------------------------------
    # Every provider call is bounded by this. A dead provider must never be able
    # to hang the gateway; it must fail fast enough for failover to be useful.
    provider_timeout_seconds: float = 30.0
    provider_connect_timeout_seconds: float = 5.0

    # --- semantic cache ------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Set to "hash" to run without downloading a model (lexical fallback).
    embedding_backend: str = "fastembed"
    cache_ttl_seconds: int = 24 * 3600
    # Upper bound on entries compared per lookup on the SQLite path.
    cache_max_candidates: int = 500

    # --- admin plane ---------------------------------------------------------
    admin_token: str = "prism-admin-token"

    # --- misc ----------------------------------------------------------------
    log_prompt_bodies: bool = False  # privacy: off by default, see docs/ARCHITECTURE.md


@lru_cache
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# Gateway (routing) configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str  # a gateway secret: never returned to callers or written to logs
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasConfig:
    """An alias such as ``fast`` — one primary model plus an ordered fallback chain."""

    alias: str
    primary: str | None
    fallbacks: tuple[str, ...] = ()
    # Only set for the ``auto`` alias: difficulty label -> the alias to delegate to.
    route_by_difficulty: dict[str, str] = field(default_factory=dict)

    @property
    def is_auto(self) -> bool:
        return bool(self.route_by_difficulty)

    @property
    def chain(self) -> tuple[str, ...]:
        """Models to try in order: primary first, then each fallback."""
        return (self.primary, *self.fallbacks) if self.primary else self.fallbacks


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_backoff_ms: int = 200
    backoff_multiplier: float = 2.0


@dataclass(frozen=True)
class DegradationConfig:
    """Thresholds for declaring a provider unhealthy from recent traffic."""

    error_rate_threshold: float = 0.5
    window_seconds: int = 60
    p95_latency_ms: int = 5000


@dataclass(frozen=True)
class GatewayConfig:
    providers: tuple[ProviderConfig, ...]
    aliases: dict[str, AliasConfig]
    retry: RetryConfig
    degradation: DegradationConfig

    def provider_for_model(self, model: str) -> ProviderConfig | None:
        """Mock and real providers alike name their models ``<provider>-<size>``."""
        for provider in self.providers:
            if model.startswith(f"{provider.name}-") or model in provider.models:
                return provider
        return None

    @classmethod
    def load(cls, path: Path, *, openrouter_api_key: str | None = None) -> GatewayConfig:
        raw = json.loads(path.read_text())

        providers = tuple(
            ProviderConfig(
                name=p["name"],
                base_url=p["base_url"],
                api_key=_provider_api_key(p, openrouter_api_key),
                models=tuple(p.get("models", ())),
            )
            for p in raw["providers"]
        )

        aliases: dict[str, AliasConfig] = {}
        for alias, spec in raw["model_aliases"].items():
            aliases[alias] = AliasConfig(
                alias=alias,
                primary=spec.get("primary"),
                fallbacks=tuple(spec.get("fallbacks", ())),
                route_by_difficulty=dict(spec.get("route_by_difficulty", {})),
            )

        retry_raw = raw.get("retry", {})
        degradation_raw = {k: v for k, v in raw.get("degradation", {}).items() if not k.startswith("_")}

        return cls(
            providers=providers,
            aliases=aliases,
            retry=RetryConfig(
                max_attempts=retry_raw.get("max_attempts", 3),
                initial_backoff_ms=retry_raw.get("initial_backoff_ms", 200),
                backoff_multiplier=retry_raw.get("backoff_multiplier", 2.0),
            ),
            degradation=DegradationConfig(**degradation_raw),
        )


def _provider_api_key(provider: dict, openrouter_api_key: str | None) -> str:
    if provider.get("api_key_env") == "OPENROUTER_API_KEY":
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when using the OpenRouter configuration.")
        return openrouter_api_key
    return provider["api_key"]


@lru_cache
def get_gateway_config() -> GatewayConfig:
    settings = get_settings()
    key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else None
    return GatewayConfig.load(settings.gateway_config_path, openrouter_api_key=key)
