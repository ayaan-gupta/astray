"""Shared test fixtures."""

import pytest

from server.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Keep monkeypatched env vars from leaking between tests via the singleton."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
