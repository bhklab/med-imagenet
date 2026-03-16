from pathlib import Path
from typing import Callable

from tqdm import tqdm

from imgnet.collections.source import (
    DropboxSource,
    S3Source,
    SourceConfig,
    TCIASource,
    ZenodoSource,
)
from imgnet.download.utils import (
    _post_unzip,
    download_file_from_s3,
    download_from_dropbox,
    download_from_zenodo,
    list_s3_bucket_keys,
)
from imgnet.loggers import logger, tqdm_logging_redirect


def _download_tcia(
    config: TCIASource,
    output_path: Path,
    series_uids: list[str],
) -> None:
    from imgnet.utils import get_idc_client

    client = get_idc_client()

    save_path = output_path
    save_path.mkdir(exist_ok=True, parents=True)
    with tqdm_logging_redirect():
        client.download_dicom_series(
            series_uids,
            save_path,
            dirTemplate="%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID",
        )


def _download_dropbox(
    config: DropboxSource, output_path: Path, dry_run: bool = False
) -> int | None:
    """If dry_run=True, return size in bytes; otherwise None."""
    if dry_run:
        return download_from_dropbox(config.url, output_path, dry_run=True)
    logger.info(f"Downloading files from Dropbox URL {config.url}")
    download_from_dropbox(config.url, output_path)
    return None


def _download_s3(
    config: S3Source, output_path: Path, dry_run: bool = False
) -> int | None:
    """If dry_run=True, return total size in bytes; otherwise None.
    If config.filenames is None, downloads all files in the bucket."""
    if config.filenames is None:
        filenames = list_s3_bucket_keys(config.bucket_name)
        if not dry_run:
            logger.info(
                f"Downloading all {len(filenames)} files from S3 bucket {config.bucket_name}"
            )
    else:
        filenames = config.filenames
    if dry_run:
        total = 0
        for file_name in filenames:
            size = download_file_from_s3(
                config.bucket_name, file_name, output_path, dry_run=True
            )
            total += size or 0
        return total
    for file_name in filenames:
        logger.info(
            f"Downloading file '{file_name}' from S3 bucket {config.bucket_name}"
        )
        download_file_from_s3(config.bucket_name, file_name, output_path)
    return None


def _download_zenodo(
    config: ZenodoSource, output_path: Path, dry_run: bool = False
) -> int | None:
    """If dry_run=True, return total size in bytes; otherwise None."""
    if dry_run:
        return download_from_zenodo(
            config.record_id,
            output_path,
            filenames=config.filenames,
            dry_run=True,
        )
    if config.filenames is None:
        logger.info(
            f"Downloading all files from Zenodo record {config.record_id}"
        )
        download_from_zenodo(config.record_id, output_path)
    else:
        for filename in config.filenames:
            logger.info(
                f"Downloading file '{filename}' from Zenodo record {config.record_id}"
            )
            download_from_zenodo(
                config.record_id, output_path, filename=filename
            )
    return None


def get_collection_download_size_bytes(config: SourceConfig) -> float:
    """Return total download size in bytes for non-TCIA sources; 0 for TCIA (use IDC for size)."""
    dummy_path = Path()
    match config:
        case TCIASource():
            return 0.0
        case DropboxSource():
            out = _download_dropbox(config, dummy_path, dry_run=True)
            return out or 0.0
        case S3Source():
            out = _download_s3(config, dummy_path, dry_run=True)
            return out or 0.0
        case ZenodoSource():
            out = _download_zenodo(config, dummy_path, dry_run=True)
            return out or 0.0


# ---- post-download steps ----

_POST_STEP_MAP: dict[str, Callable[[Path], None]] = {
    "unzip": _post_unzip,
}


def _run_post_steps(steps: list[str], output_path: Path) -> None:
    for step in steps:
        handler = _POST_STEP_MAP.get(step)
        if handler is None:
            msg = f"Unknown post_download step: '{step}'"
            raise ValueError(msg)
        handler(output_path)


def _remove_unwanted_files(output_path: Path, instance_ids: list[str]) -> None:
    instance_ids_set = set(instance_ids)
    for file in tqdm(output_path.rglob("*"), desc="Deleting files"):
        if file.is_file():
            relative_path = file.relative_to(output_path)
            if str(relative_path) not in instance_ids_set:
                file.unlink()

    logger.info(f"Deleting empty directories for {output_path}")
    for dirpath in sorted(
        (p for p in output_path.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if not any(dirpath.iterdir()):
            dirpath.rmdir()


# ---- main entry point ----


def download_collection(
    output_path: Path,
    config: SourceConfig,
    instance_ids: list[str] | None = None,
) -> None:
    """Download a collection using the transport specified in its ``source.json``.

    Parameters
    ----------
    output_path : Path
        Directory to save downloaded files into.
    config : SourceConfig
        The source configuration.
    instance_ids : list[str] | None
        The specific instance IDs to download.
    """
    if instance_ids is None:
        instance_ids = []
    output_path.mkdir(parents=True, exist_ok=True)

    match config:
        case TCIASource():
            if not instance_ids:
                raise ValueError(
                    "instance_ids(series UIDs) are required for TCIA downloads"
                )
            _download_tcia(config, output_path, series_uids=instance_ids)
        case DropboxSource():
            _download_dropbox(config, output_path, dry_run=False)
        case S3Source():
            _download_s3(config, output_path, dry_run=False)
        case ZenodoSource():
            _download_zenodo(config, output_path, dry_run=False)

    if config.post_download:
        _run_post_steps(config.post_download, output_path)

    if config.source != "tcia" and instance_ids:
        # For non-TCIA sources, we need to download the entire collection.
        # Then delete the instances that are not in the instance_ids list.
        # The instance_ids list is a list of relative paths to the files in the output_path.
        logger.info(
            f"Deleting files that are not in the instance_ids list for {output_path}"
        )
        _remove_unwanted_files(output_path, instance_ids)
