"""Unit tests for imgnet.utils.idc (global IDC client)."""

import pytest

from imgnet.utils.idc import get_idc_client, set_idc_client


@pytest.fixture(autouse=True)
def reset_idc_client():
    """Reset the global IDC client before and after each test for isolation."""
    set_idc_client(None)
    yield
    set_idc_client(None)


def test_get_idc_client_returns_same_instance():
    """First call creates a client; second call returns the same instance."""
    client1 = get_idc_client()
    client2 = get_idc_client()
    assert client1 is client2


def test_set_idc_client_none_resets_singleton():
    """After set_idc_client(None), next get_idc_client() creates a new client."""
    client1 = get_idc_client()
    set_idc_client(None)
    client2 = get_idc_client()
    assert client1 is not client2


def test_set_idc_client_custom_returns_custom():
    """set_idc_client(custom) makes get_idc_client() return that instance."""
    fake = object()
    set_idc_client(fake)  # type: ignore[arg-type]
    assert get_idc_client() is fake


def test_set_idc_client_none_then_custom():
    """Reset then set custom client; get_idc_client returns the custom one."""
    set_idc_client(None)
    fake = object()
    set_idc_client(fake)  # type: ignore[arg-type]
    assert get_idc_client() is fake


def test_set_idc_client_custom_then_reset_then_new():
    """Custom client can be replaced by reset; next get creates a new real client."""
    fake = object()
    set_idc_client(fake)  # type: ignore[arg-type]
    assert get_idc_client() is fake
    set_idc_client(None)
    client = get_idc_client()
    assert client is not fake
