from pathlib import Path, PurePosixPath

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from imgnet.utils.archive import RemoteArchive

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
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:  # noqa: SIM117
            with out_file.open("wb") as out:
                for chunk in response.iter_content(chunk_size=2**20):
                    out.write(chunk)
                    pbar.update(len(chunk))


def _contains_subpath(path: str, subpath: PurePosixPath) -> bool:
    parts = PurePosixPath(path).parts
    needle = subpath.parts
    if not needle or len(needle) > len(parts):
        return False
    span = len(needle)
    return any(
        parts[i : i + span] == needle for i in range(len(parts) - span + 1)
    )


def matching_instance_ids(
    archive_path: str, instance_ids: set[str]
) -> set[str]:
    """Return ids that sit under the extensionless archive path."""
    archive_id = PurePosixPath(RemoteArchive.strip_extension(archive_path))
    return {
        instance_id
        for instance_id in instance_ids
        if _contains_subpath(instance_id, archive_id)
    }


def archive_instance_ids(
    archive_path: str, inner_members: list[str]
) -> list[str]:
    """Map archive members to extensionless instance ids.

    ``root/zip/archive.zip`` with member ``zip_test_file.nii.gz`` becomes
    ``root/zip/archive/zip_test_file.nii.gz``. Listings that already start
    with the archive stem are joined onto the parent directory.
    """
    prefix = PurePosixPath(RemoteArchive.strip_extension(archive_path))
    stem = prefix.name
    parent = prefix.parent
    instance_ids: list[str] = []
    for member in inner_members:
        member_path = PurePosixPath(member)
        if member_path.parts and member_path.parts[0] == stem:
            instance_ids.append((parent / member_path).as_posix())
        else:
            instance_ids.append((prefix / member).as_posix())
    return instance_ids
