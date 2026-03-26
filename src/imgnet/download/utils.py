from pathlib import Path

import requests
from tqdm import tqdm


def _download_http_file(
    url: str, out_file: Path, desc: str, size: int | float | None = None
) -> None:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total = int(size or response.headers.get("content-length", 0)) # type: ignore
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:  # noqa: SIM117
            with out_file.open("wb") as out:
                for chunk in response.iter_content(chunk_size=2**20):
                    out.write(chunk)
                    pbar.update(len(chunk))
