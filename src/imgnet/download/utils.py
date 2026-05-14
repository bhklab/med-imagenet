from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

_TIMEOUT = (10, 30)  # (connect, read) in seconds
_RETRY = Retry(
    total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
)


def _http_session() -> requests.Session:
    """Return a requests Session with retry and timeout defaults."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_http_file(
    url: str, out_file: Path, desc: str, size: int | float | None = None
) -> None:
    session = _http_session()
    with session.get(url, stream=True, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        total = int(size or response.headers.get("content-length", 0))  # type: ignore
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:  # noqa: SIM117
            with out_file.open("wb") as out:
                for chunk in response.iter_content(chunk_size=2**20):
                    out.write(chunk)
                    pbar.update(len(chunk))
