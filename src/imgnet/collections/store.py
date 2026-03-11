import json
import os
from pathlib import Path

import pandas as pd
from platformdirs import user_data_dir
from rich import print
from rich.table import Table
from tqdm import tqdm

from imgnet.collections.source import (
    FileType,
    SourceConfig,
    TCIASource,
    source_adapter,
)
from imgnet.download.utils import _fetch_collection_size_idc
from imgnet.download.dispatcher import get_collection_download_size_bytes
from imgnet.loggers import logger, tqdm_logging_redirect

_ENV_VAR = "IMGNET_INDEX_DIR"
_APP_NAME = "med-imagenet"
_APP_AUTHOR = "bhklab"


def default_indexed_datasets_path() -> Path:
    """Return the default path for the ``indexed_datasets/`` directory.

    Resolution order:
    1. ``$IMGNET_INDEX_DIR/indexed_datasets`` if the env var is set.
    2. ``<platform-data-dir>/med-imagenet/indexed_datasets`` via *platformdirs*.
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env) / "indexed_datasets"
    return Path(user_data_dir(_APP_NAME, _APP_AUTHOR)) / "indexed_datasets"


class IndexedDatasets:
    """Read-only accessor for the indexed_datasets directory.

    Parameters
    ----------
    path : Path | str | None
        Path to the ``indexed_datasets`` directory
        If not provided, the latest release from Hugging Face will be downloaded.
    force_download : bool
        If true, the indexed datasets will be downloaded even if they already exist.
    """

    # ---- core data access ----

    def __init__(self, path: Path | str | None = None, force_download: bool = False) -> None:
        if path is None:
            path = default_indexed_datasets_path()

        path = Path(path)
        print(path.resolve())

        if not path.exists() or force_download:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
            from tqdm.auto import tqdm as _tqdm

            repo_id = "BruhJosh/med-image-index"
            logger.warning(
                "Indexed datasets not found at %s. "
                "Downloading latest release from Hugging Face.",
                path.resolve(),
            )
            download_dir = path.parent
            download_dir.mkdir(parents=True, exist_ok=True)

            disable_progress_bars()
            try:
                with tqdm_logging_redirect():
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        local_dir=download_dir,
                        ignore_patterns=[".git*"],
                        tqdm_class=_tqdm,
                    )
            finally:
                enable_progress_bars()

        self.path = path

    @property
    def imgtools_path(self) -> Path:
        return self.path / ".imgtools"

    @property
    def summary_path(self) -> Path:
        return self.path / "collections_summary.json"

    @property
    def collections(self) -> list[str]:
        """Collection names derived from subdirectories of ``.imgtools/``."""
        return sorted(
            folder.name
            for folder in self.imgtools_path.iterdir()
            if folder.is_dir()
        )

    def crawl_db(self, collection: str) -> dict:
        """Return the parsed ``crawl_db.json`` for *collection*."""
        db_path = self.imgtools_path / collection / "crawl_db.json"

        if not db_path.exists():
            logger.warning(f"Crawl db not found for collection {collection}. Returning empty dictionary.")
            return {}

        with open(db_path, "r") as f:
            return json.load(f)

    def index(self, collection: str) -> pd.DataFrame:
        """Return the ``index.csv`` for *collection* as a DataFrame."""
        csv_path = self.imgtools_path / collection / "index.csv"
        return pd.read_csv(csv_path)

    def source_config(self, collection: str) -> SourceConfig:
        """Return the validated ``source.json`` for *collection*.

        Falls back to ``TCIASource()`` (DICOM/TCIA defaults) when no
        ``source.json`` exists, keeping backwards compatibility with
        collections that predate this file.
        """
        config_path = self.imgtools_path / collection / "source.json"
        if not config_path.exists():
            return TCIASource()
        with open(config_path, "r") as f:
            return source_adapter.validate_python(json.load(f))

    def file_type(self, collection: str) -> FileType:
        """Return the ``FileType`` for *collection*."""
        return self.source_config(collection).file_type

    def collection_size(self, collection: str) -> float:
        """Return the size of *collection* in GB."""
        config = self.source_config(collection)

        try:
            if config.source == "tcia":
                return _fetch_collection_size_idc(collection)
            return round(float(get_collection_download_size_bytes(config) / 1e9), 2)
        except Exception as e:
            logger.error(f"Error getting size for collection {collection}: {e}")
            return 0.0

    # ---- summary / display ----

    def summary(self, update: bool = False) -> dict:
        """Parsed ``collections_summary.json``, or ``None`` if it doesn't exist."""
        if not self.summary_path.exists() or update:
            logger.info("Collections summary not found or update is True. Building new summary.")
            collection_db = self._build_collection_db()
            with open(self.summary_path, "w") as f:
                json.dump(collection_db, f)
            return collection_db
        with open(self.summary_path, "r") as f:
            logger.info("Loading collections summary from %s.", self.summary_path)
            return json.load(f)

    def _build_collection_db(self) -> dict:
        collection_db = {}

        with tqdm_logging_redirect():
            for collection in tqdm(self.collections, desc="Building collections summary", total=len(self.collections)):
                logger.info("Building collection summary for %s.", collection)
                summary = {
                    "Modalities": set(),
                    "BodyPartsExamined": set(),
                    "SeriesCount": 0,
                    "Size": self.collection_size(collection)
                }

                if self.file_type(collection) == FileType.NIFTI:
                    index = self.index(collection)
                    if "Modality" in index.columns:
                        summary["Modalities"] = index["Modality"].dropna().unique().tolist()
                    else:
                        summary["Modalities"] = []
                    if "BodyPartExamined" in index.columns:
                        summary["BodyPartsExamined"] = index["BodyPartExamined"].dropna().unique().tolist()
                    else:
                        summary["BodyPartsExamined"] = []
                    if "SeriesInstanceUID" in index.columns:
                        summary["SeriesCount"] = int(index["SeriesInstanceUID"].nunique())
                    else:
                        summary["SeriesCount"] = int(index["reference_id"].nunique())

                elif self.file_type(collection) == FileType.DICOM:

                    crawl_json = self.crawl_db(collection)
                    for key in crawl_json:
                        series = crawl_json[key][list(crawl_json[key].keys())[0]]
                        if series["Modality"]:
                            summary["Modalities"].add(series["Modality"])
                        if series["BodyPartExamined"]:
                            summary["BodyPartsExamined"].add(series["BodyPartExamined"])
                        summary["SeriesCount"] += 1
                    for key in summary:
                        if isinstance(summary[key], set):
                            summary[key] = list(summary[key])
                    
                collection_db[collection] = summary

        return collection_db

    def display_summary(self, update: bool = False) -> None:
        table = Table(title="Collections Summary")
        table.add_column("Collection", justify="left")
        table.add_column("BodyPartsExamined", justify="left")
        table.add_column("Modalities", justify="left")
        table.add_column("Series Count", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("File Type", justify="left")
        table.add_column("Source", justify="left")

        collection_db = self.summary(update)

        for collection, info in collection_db.items():
            table.add_row(
                collection,
                ", ".join(info["BodyPartsExamined"]),
                ", ".join(info["Modalities"]),
                f"{info['SeriesCount']}",
                f"{info['Size']} GB",
                f"{self.file_type(collection).value.upper()}",
                f"{self.source_config(collection).source.upper()}",
            )

        print(table)

if __name__ == "__main__":
    store = IndexedDatasets("/home/joshua-siraj/Documents/BHKLAB/med-image-index/indexed_datasets")
    store.display_summary(update=True)

    