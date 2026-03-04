import shutil
from pathlib import Path
from typing import Callable

from idc_index import IDCClient

from imgnet.collections.source import (
    DropboxSource,
    S3Source,
    TCIASource,
    ZenodoSource,
)
from imgnet.collections.store import IndexedDatasets
from imgnet.download.utils import (
    download_file_from_s3,
    download_from_dropbox,
    download_from_zenodo,
)
from imgnet.loggers import logger, tqdm_logging_redirect


def _download_tcia(
    config: TCIASource,
    output_path: Path,
    client: IDCClient | None = None,
    series_uids: list[str] | None = None,
) -> None:
    if client is None:
        raise ValueError("IDCClient is required for TCIA downloads.")
    for uid in series_uids or []:
        logger.info(f"Downloading DICOM series {uid}")
        save_path = output_path / uid
        save_path.mkdir(exist_ok=True, parents=True)
        with tqdm_logging_redirect():
            client.download_dicom_series(uid, save_path)


def _download_dropbox(config: DropboxSource, output_path: Path) -> None:
    download_from_dropbox(config.url, output_path)


def _download_s3(config: S3Source, output_path: Path) -> None:
    download_file_from_s3(config.bucket_name, config.file_name, output_path)


def _download_zenodo(config: ZenodoSource, output_path: Path) -> None:
    download_from_zenodo(config.record_id, output_path, filename=config.filename)


# ---- post-download steps ----

def _post_unzip(output_path: Path) -> None:
    """Unpack every archive found in *output_path*."""
    for archive in output_path.iterdir():
        if archive.suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            logger.info(f"Unpacking {archive.name}")
            shutil.unpack_archive(archive, output_path)
            archive.unlink()


_POST_STEP_MAP: dict[str, Callable[[Path], None]] = {
    "unzip": _post_unzip,
}


def _run_post_steps(steps: list[str], output_path: Path) -> None:
    for step in steps:
        handler = _POST_STEP_MAP.get(step)
        if handler is None:
            raise ValueError(f"Unknown post_download step: '{step}'")
        handler(output_path)


# ---- main entry point ----

def download_collection(
    collection: str,
    output_path: Path,
    store: IndexedDatasets,
    client: IDCClient | None = None,
    series_uids: list[str] | None = None,
) -> None:
    """Download a collection using the transport specified in its ``source.json``.

    Parameters
    ----------
    collection : str
        Name of the collection to download.
    output_path : Path
        Directory to save downloaded files into.
    store : IndexedDatasets
        The indexed datasets store (used to read ``source.json``).
    client : IDCClient | None
        Required for TCIA/DICOM downloads. Ignored for other sources.
    series_uids : list[str] | None
        For TCIA downloads, the specific series UIDs to download.
        Ignored for non-TCIA sources (which download the entire collection).
    """
    config = store.source_config(collection)
    output_path.mkdir(exist_ok=True, parents=True)

    match config:
        case TCIASource():
            _download_tcia(config, output_path, client=client, series_uids=series_uids)
        case DropboxSource():
            _download_dropbox(config, output_path)
        case S3Source():
            _download_s3(config, output_path)
        case ZenodoSource():
            _download_zenodo(config, output_path)

    if config.post_download:
        _run_post_steps(config.post_download, output_path)
