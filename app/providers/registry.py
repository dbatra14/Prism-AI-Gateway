"""Builds the live provider adapters from the gateway config.

One process-wide registry, created at startup and closed at shutdown, so HTTP
connections are pooled and reused across requests rather than re-established per
call. Swapping mock providers for real ones is a config edit, not a code change.
"""

from __future__ import annotations

from app.config import GatewayConfig, Settings
from app.providers.base import Provider
from app.providers.openai_compat import OpenAICompatibleProvider


class ProviderRegistry:
    def __init__(self, providers: dict[str, Provider]):
        self._providers = providers

    @classmethod
    def from_config(cls, config: GatewayConfig, settings: Settings) -> ProviderRegistry:
        providers: dict[str, Provider] = {
            spec.name: OpenAICompatibleProvider(
                name=spec.name,
                base_url=spec.base_url,
                api_key=spec.api_key,
                timeout_seconds=settings.provider_timeout_seconds,
                connect_timeout_seconds=settings.provider_connect_timeout_seconds,
            )
            for spec in config.providers
        }
        return cls(providers)

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def names(self) -> list[str]:
        return list(self._providers)

    def all(self) -> list[Provider]:
        return list(self._providers.values())

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
