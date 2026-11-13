"""The gateway's error taxonomy.

Every rejection a caller can hit is one of these, and each carries a distinct
``type`` so a client can tell *why* it was rejected from the body alone —
"429" by itself doesn't distinguish "you're going too fast" from "you're out of
money", and those need opposite responses from the caller.

``log_status`` is what gets written to the request log, so the ops console can
show rejections broken down by reason.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import TimeoutError as SQLTimeoutError


class PrismError(Exception):
    """Base class for every caller-visible rejection."""

    status_code: int = 500
    error_type: str = "internal_error"
    log_status: str = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_body(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.error_type,
            }
        }

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.to_body())


class AuthenticationError(PrismError):
    """Missing, malformed, unknown, or disabled virtual key."""

    status_code = 401
    error_type = "authentication_error"
    log_status = "rejected_auth"


class ModelNotAllowedError(PrismError):
    """The key is valid but this model is not on its allowlist."""

    status_code = 403
    error_type = "model_not_allowed"
    log_status = "rejected_allowlist"


class RateLimitError(PrismError):
    """Requests-per-minute limit exhausted for the current window."""

    status_code = 429
    error_type = "rate_limit_exceeded"
    log_status = "rejected_rate_limit"


class BudgetExceededError(PrismError):
    """Monthly spend cap reached. Distinct from a rate limit: waiting won't help."""

    status_code = 402
    error_type = "budget_exceeded"
    log_status = "rejected_budget"


class ModelNotFoundError(PrismError):
    """The requested alias or model isn't configured on this gateway."""

    status_code = 404
    error_type = "not_found_error"
    log_status = "rejected_unknown_model"


class InvalidRequestError(PrismError):
    """Malformed body — e.g. an empty ``messages`` array."""

    status_code = 400
    error_type = "invalid_request_error"
    log_status = "rejected_invalid"


class UpstreamError(PrismError):
    """Every provider in the chain failed, after retries and failover."""

    status_code = 502
    error_type = "upstream_error"
    log_status = "upstream_error"


class ServiceOverloadedError(PrismError):
    """The gateway itself is out of capacity — e.g. no free database connection.

    Distinct from ``UpstreamError``: nothing is wrong with the providers, the
    gateway is the bottleneck. Callers should back off and retry.
    """

    status_code = 503
    error_type = "service_overloaded"
    log_status = "gateway_overloaded"


async def prism_error_handler(_request: Request, exc: PrismError) -> JSONResponse:
    return exc.to_response()


async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Last resort: keep the error contract even for bugs.

    A caller of an OpenAI-compatible endpoint parses OpenAI-shaped errors. If an
    unexpected exception escaped as FastAPI's default ``{"detail": ...}``, a
    client's error handling would break on the one response it most needs to
    understand. The internal detail is logged, not returned.
    """
    import logging

    logging.getLogger(__name__).exception("unhandled error: %s", exc)

    # Pool exhaustion is a capacity problem, not a bug — say so, and say it fast.
    if isinstance(exc, SQLTimeoutError):
        return ServiceOverloadedError(
            "The gateway is at capacity and could not obtain a database connection. Retry shortly."
        ).to_response()

    return PrismError("The gateway encountered an internal error.").to_response()
