import pytest

from idc_index import IDCClient
from imgnet.collections.store import IndexedDatasets

@pytest.fixture(scope="session")
def client():
    return IDCClient()

@pytest.fixture(scope="session")
def store():
    return IndexedDatasets()

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