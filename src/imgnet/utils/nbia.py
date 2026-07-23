"""Global NBIA client used by the package for private TCIA index and downloads."""

import threading

import os
from dotenv import load_dotenv
from tcia_utils import nbia

class NBIAClientWrapper:
    """Wrapper around tcia_utils to manage authentication."""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._authenticated = False
    
    def authenticate(self):
        """Authenticate with TCIA."""
        if not self._authenticated:
            nbia.getToken(username=self.username, password=self.password)
            self._authenticated = True
    
    def downloadSeries(self, series_uids: list[str], path: str):
        """Download series using the authenticated session."""
        self.authenticate()
        return nbia.downloadSeries(series_uids, path=path)
    
    def getSeries(self, collection: str):
            """Get series metadata for each series in a collection"""
            self.authenticate()
            return nbia.getSeries(collection=collection)
    



_state: dict[str, NBIAClientWrapper | None] = {"client": None}
_lock = threading.Lock()


def get_nbia_client() -> NBIAClientWrapper:
    """Return the shared IDC client, creating it on first use. Thread-safe."""
    if _state["client"] is None:
        with _lock:
            if _state["client"] is None:
                _state["client"] = NBIAClientWrapper(username=os.getenv('NBIA_USERNAME'), password=os.getenv('NBIA_PASSWORD'))

    if _state["client"] is None:
        raise RuntimeError("Failed to create NBIA client")
    return _state["client"]


def set_nbia_client(client: NBIAClientWrapper | None) -> None:
    """Set the global IDC client (e.g. for tests or custom config). Pass None to reset."""
    with _lock:
        _state["client"] = client
