import functools
from pathlib import Path

import orjson
import pandas as pd
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm

from imgnet.collections.source import (
    FileType,
    SourceConfig,
    TCIASource,
    source_adapter,
)
from imgnet.collections.utils import (
    _default_indexed_datasets_path,
    _fetch_collection_description_tcia,
)
from imgnet.download.dispatcher import get_collection_download_size_bytes
from imgnet.download.utils import _fetch_collection_size_idc
from imgnet.loggers import logger, tqdm_logging_redirect


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

    def __init__(
        self, path: Path | str | None = None, force_download: bool = False
    ) -> None:
        if path is None:
            path = _default_indexed_datasets_path()

        path = Path(path)
        logger.info(f"Indexed datasets path: {path.resolve()}")

        if not path.exists() or force_download:
            from huggingface_hub import list_repo_commits, snapshot_download
            from huggingface_hub.utils import (
                disable_progress_bars,
                enable_progress_bars,
            )
            from tqdm.auto import tqdm as _tqdm

            repo_id = "BruhJosh/med-image-index"
            latest_commit = list_repo_commits(
                repo_id=repo_id, repo_type="dataset"
            )[0].title
            logger.warning(
                "Indexed datasets not found at %s. "
                "Downloading latest release from Hugging Face. Latest commit: %s",
                path.resolve(),
                latest_commit,
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
        self._collection_cache: dict[str, "Collection"] = {}

    @property
    def imgtools_path(self) -> Path:
        return self.path / ".imgtools"

    @property
    def summary_path(self) -> Path:
        return self.path / "collections_summary.json"

    @functools.cached_property
    def collections(self) -> list[str]:
        """Collection names derived from subdirectories of ``.imgtools/``."""
        return sorted(
            folder.name
            for folder in self.imgtools_path.iterdir()
            if folder.is_dir()
        )

    def get_collection(self, name: str) -> "Collection":
        """Return a cached Collection for the given name. Validates that the collection exists."""
        if name not in self.collections:
            error_message = (
                f"Unknown collection: {name!r}. Known: {self.collections}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        if name not in self._collection_cache:
            self._collection_cache[name] = Collection(
                name=name, path=self.imgtools_path / name
            )
        return self._collection_cache[name]

    def crawl_db(self, collection: str) -> dict:
        """Return the parsed ``crawl_db.json`` for *collection*."""
        return self.get_collection(collection).crawl_db

    def index(self, collection: str) -> pd.DataFrame:
        """Return the ``index.csv`` for *collection* as a DataFrame."""
        return self.get_collection(collection).index

    def source_config(self, collection: str) -> SourceConfig:
        """Return the validated ``source.json`` for *collection*.

        Falls back to ``TCIASource()`` (DICOM/TCIA defaults) when no
        ``source.json`` exists, keeping backwards compatibility with
        collections that predate this file.
        """
        return self.get_collection(collection).source_config

    def file_type(self, collection: str) -> FileType:
        """Return the ``FileType`` for *collection*."""
        return self.get_collection(collection).file_type

    def collection_size(self, collection: str) -> float:
        """Return the size of *collection* in GB."""
        return self.get_collection(collection).collection_size

    def description(self, collection: str) -> str:
        """Return the description of *collection*."""
        return self.get_collection(collection).description

    def supported_query_tags(self, collection: str) -> dict[str, list[str]]:
        """Return supported query tags per modality for *collection*."""
        return self.get_collection(collection).supported_query_tags

    def display_supported_query_tags(self, collection: str) -> None:
        """Display supported query tags per modality for *collection*."""
        supported_tags = self.supported_query_tags(collection)
        table = Table(title=f"Supported Query Tags for {collection}")
        table.add_column("Modality", justify="left")
        table.add_column("Supported Query Tags", justify="left")
        first = True
        for modality, tags in supported_tags.items():
            if not first:
                table.add_row("", "")  # Add a blank line between rows
            table.add_row(modality, ", ".join(tags))
            first = False
        rprint(table)

    # ---- summary / display ----

    def summary(self, update: bool = False) -> dict:
        """Parsed ``collections_summary.json``, or ``None`` if it doesn't exist."""
        if not self.summary_path.exists() or update:
            logger.info(
                "Collections summary not found or update is True. Building new summary."
            )
            collection_db = self._build_collection_db()
            with self.summary_path.open("wb") as f:
                f.write(orjson.dumps(collection_db))
            return collection_db
        logger.info("Loading collections summary from %s.", self.summary_path)
        return orjson.loads(self.summary_path.read_bytes())

    def _build_collection_db(self) -> dict:
        collection_db = {}
        with tqdm_logging_redirect():
            for collection in tqdm(
                self.collections,
                desc="Building collections summary",
                total=len(self.collections),
            ):
                logger.info("Building collection summary for %s.", collection)
                collection_db[collection] = self.get_collection(
                    collection
                ).build_summary_entry()
        return collection_db

    def display_summary(self, update: bool = False) -> None:
        table = Table(title="Collections Summary")
        table.add_column("Collection", justify="left")
        table.add_column("BodyPartsExamined", justify="left")
        table.add_column("Modalities", justify="left")
        table.add_column("Images", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("File Type", justify="left")
        table.add_column("Source", justify="left")
        collection_db = self.summary(update)

        for collection, info in collection_db.items():
            table.add_row(
                collection,
                ", ".join(info["BodyPartsExamined"]),
                ", ".join(info["Modalities"]),
                f"{info['Images']}",
                f"{info['Size']} GB",
                f"{info['File Type']}",
                f"{info['Source']}",
            )

        rprint(table)


class Collection:
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.indexed_datasets_path = path.parent.parent

    @functools.cached_property
    def index(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "index.csv")

    @functools.cached_property
    def crawl_db(self) -> dict:
        db_path = self.path / "crawl_db.json"
        if not db_path.exists():
            logger.warning(
                f"Crawl db not found for collection {self.name}. Returning empty dictionary."
            )
            return {}
        if self.file_type != FileType.DICOM:
            logger.warning(
                f"Crawl db not supported for collection {self.name} of type {self.file_type}. Returning empty dictionary."
            )
            return {}
        return orjson.loads(db_path.read_bytes())

    @functools.cached_property
    def source_config(self) -> SourceConfig:
        """Return the validated source config. Falls back to TCIASource() when source.json is missing."""
        config_path = self.path / "source.json"
        if not config_path.exists():
            return TCIASource()
        return source_adapter.validate_python(
            orjson.loads(config_path.read_bytes())
        )

    @property
    def file_type(self) -> FileType:
        return self.source_config.file_type

    @functools.cached_property
    def summary(self) -> dict:
        return orjson.loads(
            (
                self.indexed_datasets_path / "collections_summary.json"
            ).read_bytes()
        )[self.name]

    @functools.cached_property
    def collection_size(self) -> float:
        config = self.source_config

        try:
            if config.source == "tcia":
                return _fetch_collection_size_idc(self.name)
            return round(
                float(get_collection_download_size_bytes(config) / 1e9), 2
            )
        except Exception as e:
            logger.error(f"Error getting size for collection {self.name}: {e}")
            return 0.0

    @functools.cached_property
    def description(self) -> str:
        """Return the description of *collection*."""
        try:
            if self.source_config.source == "tcia":
                return _fetch_collection_description_tcia(self.name)
            else:
                return self.source_config.description
        except Exception as e:
            logger.error(
                f"Error getting description for collection {self.name}: {e}"
            )
            return ""

    @functools.cached_property
    def supported_query_tags(self) -> dict[str, list[str]]:
        modalities = self.summary["Modalities"]

        supported_tags = dict()
        for modality in modalities:
            supported_tags[modality] = set()

        index = self.index
        for modality in modalities:
            subset = (
                index[index["Modality"] == modality]
                .dropna(axis=1, how="all")
                .dropna()
            )
            supported_tags[modality].update(subset.columns.tolist())

        if self.file_type == FileType.DICOM:
            crawl_db = self.crawl_db
            for modality in modalities:
                for key in crawl_db:
                    series = crawl_db[key][list(crawl_db[key].keys())[0]]
                    if series["Modality"] == modality:
                        supported_tags[modality].update(list(series.keys()))

        return {
            modality: sorted(list(tags))
            for modality, tags in supported_tags.items()
        }

    def build_summary_entry(self) -> dict:
        """Build the summary dict for this collection (Modalities, BodyPartsExamined, Images, Size, etc.)."""
        summary = {
            "Modalities": set(),
            "BodyPartsExamined": set(),
            "Images": len(self.index),
            "Size": self.collection_size,
            "File Type": self.file_type.value.upper(),
            "Source": self.source_config.source.upper(),
        }
        if self.file_type == FileType.NIFTI:
            index = self.index
            if "Modality" in index.columns:
                summary["Modalities"] = (
                    index["Modality"].dropna().unique().tolist()
                )
            else:
                summary["Modalities"] = []
            if "BodyPartExamined" in index.columns:
                summary["BodyPartsExamined"] = (
                    index["BodyPartExamined"].dropna().unique().tolist()
                )
            else:
                summary["BodyPartsExamined"] = []
        elif self.file_type == FileType.DICOM:
            crawl_json = self.crawl_db
            for _, value in crawl_json.items():
                series = value[next(iter(value))]
                if series.get("Modality"):
                    summary["Modalities"].add(series["Modality"])
                if series.get("BodyPartExamined"):
                    summary["BodyPartsExamined"].add(
                        series["BodyPartExamined"]
                    )
            for key, value in summary.items():
                if isinstance(value, set):
                    summary[key] = list(value)
        return summary
