"""Request and response shapes for the data plane.

The request model is deliberately permissive about *extra* fields: unknown keys
(``temperature``, ``max_tokens``, ``top_p``, ...) are collected and forwarded
upstream untouched. A gateway that only passes through the parameters it was
taught about would silently drop features the moment a provider adds one, and
callers would have no way to tell.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Fields the gateway consumes itself and must not forward upstream verbatim.
_GATEWAY_OWNED = {"model", "messages", "stream"}


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class ChatCompletionRequest(BaseModel):
    """An OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="A model alias (fast/smart/auto) or a concrete model name")
    messages: list[Message]
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, value: list[Message]) -> list[Message]:
        if not value:
            raise ValueError("'messages' must contain at least one message")
        return value

    def messages_as_dicts(self) -> list[dict]:
        return [m.model_dump(exclude_none=False) for m in self.messages]

    def passthrough_params(self) -> dict:
        """Everything the caller sent that belongs to the provider, not to us."""
        return {
            key: value
            for key, value in self.model_dump(exclude_unset=True).items()
            if key not in _GATEWAY_OWNED
        }


class UsageSummary(BaseModel):
    key: str
    from_: str = Field(..., alias="from")
    to: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hits: int

    model_config = ConfigDict(populate_by_name=True)


class CacheStats(BaseModel):
    key: str | None = None
    hits: int
    misses: int
    hit_rate: float
    entries: int


class HealthStatus(BaseModel):
    provider: str
    reachable: bool
    healthy: bool
    requests_in_window: int
    error_rate: float
    p95_latency_ms: int
    status: Literal["healthy", "degraded", "unreachable"]
