"""Access to the process-wide singletons built at startup.

The provider registry, dispatcher, price table and embedder are expensive to
build (HTTP pools, an ONNX model) and must not be rebuilt per request. They live
on ``app.state``; these helpers are how request handlers reach them.
"""

from __future__ import annotations

from fastapi import Request

from app.cache.semantic_cache import SemanticCache
from app.config import GatewayConfig, Settings
from app.dispatch import Dispatcher
from app.pricing import PriceTable
from app.providers.registry import ProviderRegistry


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher


def get_registry(request: Request) -> ProviderRegistry:
    return request.app.state.registry


def get_prices(request: Request) -> PriceTable:
    return request.app.state.prices


def get_cache(request: Request) -> SemanticCache:
    return request.app.state.cache


def get_config(request: Request) -> GatewayConfig:
    return request.app.state.gateway_config


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings
