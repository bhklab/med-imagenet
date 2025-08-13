import pytest
from rich.live import Live

from nbiatoolkit.nbia import NBIAClient

@pytest.fixture(scope="module")
def client():
    return NBIAClient()

@pytest.fixture(autouse=True)
def disable_rich_live(monkeypatch):
    class DummyLive:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
        def update(self, *args, **kwargs):
            pass

    monkeypatch.setattr("rich.live.Live", DummyLive)