"""Global IDC client used by the package for TCIA/IDC index and downloads."""

import threading

from idc_index import IDCClient

_state: dict[str, IDCClient | None] = {"client": None}
_lock = threading.Lock()


def get_idc_client() -> IDCClient:
    """Return the shared IDC client, creating it on first use. Thread-safe."""
    if _state["client"] is None:
        with _lock:
            if _state["client"] is None:
                _state["client"] = IDCClient()

    if _state["client"] is None:
        raise RuntimeError("Failed to create IDC client")
    return _state["client"]


def set_idc_client(client: IDCClient | None) -> None:
    """Set the global IDC client (e.g. for tests or custom config). Pass None to reset."""
    with _lock:
        _state["client"] = client
