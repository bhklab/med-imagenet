"""Global IDC client used by the package for TCIA/IDC index and downloads."""

import threading

from idc_index import IDCClient

_client: IDCClient | None = None
_lock = threading.Lock()


def get_idc_client() -> IDCClient:
    """Return the shared IDC client, creating it on first use. Thread-safe."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = IDCClient()
    return _client


def set_idc_client(client: IDCClient | None) -> None:
    """Set the global IDC client (e.g. for tests or custom config). Pass None to reset."""
    global _client
    with _lock:
        _client = client
