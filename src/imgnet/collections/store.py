import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.collections.utils import _fetch_collection_size

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

        if not path.exists() or force_download:
            from huggingface_hub import snapshot_download

            repo_id = "BruhJosh/med-image-index"
            logger.warning(
                "Indexed datasets not found at %s. "
                "Downloading latest release from Hugging Face.",
                path,
            )
            download_dir = path.parent
            download_dir.mkdir(parents=True, exist_ok=True)
            with tqdm_logging_redirect():
                snapshot_download(
                    repo_id=repo_id, 
                    repo_type="dataset",
                    local_dir=download_dir,
                    ignore_patterns=[".git*"]
                )

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

    # ---- summary / display ----

    @property
    def summary(self) -> dict:
        """Parsed ``collections_summary.json``, or ``None`` if it doesn't exist."""
        if not self.summary_path.exists():
            collection_db = self._build_collection_db()
            with open(self.summary_path, "w") as f:
                json.dump(collection_db, f)
            return collection_db
        with open(self.summary_path, "r") as f:
            return json.load(f)

    @property
    def collection_sizes(self) -> dict[str, str]:
        """Return the sizes of all collections."""
        sizes = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_collection_size, c): c for c in self.collections}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Scraping collections"):
                collection, size = f.result()
                sizes[collection] = size
        return sizes

    def _build_collection_db(self) -> dict:
        sizes = self.collection_sizes
        collection_db = {}
        for collection in self.collections:
            crawl_json = self.crawl_db(collection)

            summary = {
                "Modalities": set(),
                "BodyPartsExamined": set(),
                "SeriesCount": 0,
                "Size": "".join(sizes[collection])
            }
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

    def display_summary(self) -> None:
        table = Table(title="Collections Summary")
        table.add_column("Collection", justify="right")
        table.add_column("BodyPartsExamined", justify="left")
        table.add_column("Modalities", justify="left")
        table.add_column("Series Count", justify="right")
        table.add_column("Size", justify="right")

        collection_db = self.summary

        for collection, info in collection_db.items():
            table.add_row(
                collection,
                ", ".join(info["BodyPartsExamined"]),
                ", ".join(info["Modalities"]),
                f"{info['SeriesCount']}",
                info["Size"],
            )

        print(table)

if __name__ == "__main__":
    store = IndexedDatasets(force_download=True)
    print(store.collections[:10])