import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rich import print
from rich.table import Table
from tqdm import tqdm

from imgnet.collections.source import (
    FileType,
    SourceConfig,
    TCIASource,
    source_adapter,
)
from imgnet.loggers import logger
from imgnet.collections.utils import _fetch_collection_size


class IndexedDatasets:
    """Read-only accessor for the indexed_datasets directory.

    Parameters
    ----------
    path : Path | str | None
        Path to the ``indexed_datasets/`` root directory
        If not provided, the latest release from GitHub will be downloaded and unpacked.
    """

    # ---- core data access ----

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            from imgnet.download.utils import download_latest_release_asset, _post_unzip

            logger.warning("Indexed datasets directory not found. Downloading latest release from GitHub.")
            zip_path = download_latest_release_asset(
                owner="bhklab",
                repo="med-image_index",
                asset_name="indexed_datasets.tar.gz",
                download_dir=Path.cwd().as_posix(),
            )
            _post_unzip(zip_path.parent, archive_filenames=["indexed_datasets.tar.gz"])
            path = zip_path.parent / "indexed_datasets"
        self.path = Path(path)

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
