import functools
import operator
import shutil
from collections.abc import Iterable
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

from imgnet.collections.source import (
    DropboxSource,
    FileType,
    HuggingFaceSource,
    S3Source,
    SourceConfig,
    TCIASource,
    PrivateTCIASource,
    ZenodoSource,
    source_adapter,
)
from imgnet.collections.utils import (
    _default_indexed_datasets_path,
    _fetch_collection_description_tcia,
    _convert_tcia_collection_name_to_idc
)
from imgnet.download.base import BaseDownloader
from imgnet.download.downloaders import (
    DropboxDownloader,
    HuggingFaceDownloader,
    IDCDownloader,
    NBIADownloader,
    S3Downloader,
    ZenodoDownloader,
)
from imgnet.loggers import logger, tqdm_logging_redirect

from imgnet.utils import get_idc_client


def _unknown_collection_message(known: list[str], name: str) -> str:
    """Build an error message with optional fuzzy name suggestions."""
    suggestions = get_close_matches(name, known, n=5, cutoff=0.42)
    parts = [f"Unknown collection {name!r}."]
    if suggestions:
        parts.append("Did you mean: " + ", ".join(suggestions) + "?")
    else:
        parts.append("Inspect IndexedDatasets.collections for valid names.")
    return " ".join(parts)


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
            from huggingface_hub import list_repo_commits

            repo_id = "bhklab2026/med-image-index"
            latest_commit = list_repo_commits(
                repo_id=repo_id, repo_type="dataset",
            )[0].title
            logger.warning(
                "Indexed datasets not found at %s or force_download is True. "
                "Downloading latest release from Hugging Face, for repo %s. Latest commit: %s",
                path.resolve(),
                repo_id,
                latest_commit,
            )

            if path.exists():
                shutil.rmtree(path)
                logger.warning(
                    "Deleted existing indexed datasets directory at %s.",
                    path.resolve(),
                )

            download_dir = path.parent
            download_dir.mkdir(parents=True, exist_ok=True)

            downloader = HuggingFaceDownloader(repo_id)
            downloader.download(
                output_path=download_dir,
                ignore_patterns=[".git*"],
                force_download=True,
            )

        self.path = path
        self._collection_cache: dict[str, "Collection"] = {}

    @property
    def summary_path(self) -> Path:
        return self.path / "collections_summary.json"

    @property
    def collections(self) -> list[str]:
        """Collection names derived from subdirectories of ``indexed_datasets/``."""
        return sorted(
            folder.name
            for folder in self.path.iterdir()
            if folder.is_dir() and (folder / "parquet").is_dir()
        )

    def get_collection(self, name: str) -> "Collection":
        """Return a cached Collection for the given name. Validates that the collection exists."""
        if name not in self.collections:
            error_message = _unknown_collection_message(self.collections, name)
            logger.error(error_message)
            raise ValueError(error_message)
        if name not in self._collection_cache:
            self._collection_cache[name] = Collection(
                name=name, path=self.path / name
            )
        return self._collection_cache[name]

    def index(self, collection: str) -> pd.DataFrame:
        """Return the full Parquet index for *collection* as a DataFrame (all partitions)."""
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

    def downloader(self, collection: str) -> BaseDownloader:
        """Return the downloader for *collection*."""
        return self.get_collection(collection).downloader

    # ---- summary ----

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


class Collection:
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        # Parent of ``.../<collection>/`` is the indexed-datasets root (``collections_summary.json``).
        self.indexed_datasets_path = path.parent

    @property
    def parquet_root(self) -> Path:
        """Root directory of the Hive-partitioned Parquet dataset (``.../<collection>/parquet``)."""
        return self.path / "parquet"

    @property
    def _dataset(self) -> ds.Dataset:
        root = self.parquet_root
        if not root.is_dir():
            msg = f"Collection {self.name!r} has no Parquet index directory at {root}"
            raise FileNotFoundError(msg)
        return ds.dataset(str(root), format="parquet", partitioning="hive")

    def read_index_rows(
        self,
        modalities: list[str] | None = None,
        *,
        sample_ids: Iterable[Any] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load index rows from Parquet, optionally restricting Hive partitions via *modalities*.

        When *modalities* is not ``None``, applies a dataset filter on ``Modality`` so
        PyArrow can prune partition directories (e.g. ``Modality=CT``). Rows should
        include a ``Modality`` column (not only a Hive partition path) so filters bind
        to the file schema across engines.

        When *sample_ids* is set, adds a ``SampleID`` membership filter (combined with
        modality filters using logical AND). Row groups may still be skipped via Parquet
        statistics when available.

        Parameters
        ----------
        modalities
            If set, only rows (and partitions) for these DICOM modalities are read.
            An empty list reads nothing and returns an empty DataFrame.
        sample_ids
            If set, only rows whose ``SampleID`` is in this set are read.
            An empty iterable yields an empty DataFrame.
        columns
            Optional column projection passed to the dataset scanner.
        """
        if modalities is not None and len(modalities) == 0:
            return pd.DataFrame()

        sid_list: list[Any] | None = None
        if sample_ids is not None:
            sid_list = list(sample_ids)
            if len(sid_list) == 0:
                return pd.DataFrame()

        parts: list = []
        if modalities is not None:
            parts.append(ds.field("Modality").isin(modalities))
        if sid_list is not None:
            parts.append(ds.field("SampleID").isin(sid_list))

        filt = functools.reduce(operator.and_, parts) if parts else None

        scanner = self._dataset.scanner(filter=filt, columns=columns)
        table = scanner.to_table()
        return table.to_pandas()

    @property
    def index(self) -> pd.DataFrame:
        """Full index for this collection (all modalities / partitions). Expensive for large corpora."""
        return self.read_index_rows(modalities=None)

    @functools.cached_property
    def source_config(self) -> SourceConfig:
        """Return the validated source config. Falls back to TCIASource() when source.json is missing."""
        config_path = self.path / "source.json"
        if not config_path.exists():
            client = get_idc_client()
            public_collections = client.get_collections()
            if _convert_tcia_collection_name_to_idc(self.name) in public_collections:
                return TCIASource()
            else:
                return PrivateTCIASource()
        return source_adapter.validate_python(
            orjson.loads(config_path.read_bytes())
        )

    @property
    def file_type(self) -> FileType:
        return self.source_config.file_type

    @property
    def downloader(self) -> BaseDownloader:
        match self.source_config:
            case TCIASource():
                return IDCDownloader(self.name)
            case PrivateTCIASource():
                return NBIADownloader(self.name)
            case S3Source():
                return S3Downloader(self.source_config.bucket_name)
            case ZenodoSource():
                return ZenodoDownloader(self.source_config.record_id)
            case HuggingFaceSource():
                return HuggingFaceDownloader(self.source_config.repo_id)
            case DropboxSource():
                return DropboxDownloader(self.source_config.url)

    @functools.cached_property
    def summary(self) -> dict:
        return orjson.loads(
            (
                self.indexed_datasets_path / "collections_summary.json"
            ).read_bytes()
        )[self.name]

    @functools.cached_property
    def collection_size(self) -> float:
        return self.downloader.size

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

        supported_tags: dict[str, set[str]] = {}
        for modality in modalities:
            supported_tags[modality] = set()

        for modality in modalities:
            subset = (
                self.read_index_rows([modality])
                .dropna(axis=1, how="all")
                .dropna()
            )
            supported_tags[modality].update(subset.columns.tolist())

        return {modality: list(tags) for modality, tags in supported_tags.items()}

    def build_summary_entry(self) -> dict:
        """Build the summary dict for this collection (Modalities, BodyPartsExamined, Images, Size, etc.)."""
        summary = {
            "Modalities": set(),
            "BodyPartsExamined": set(),
            "Images": self._dataset.count_rows(),
            "Size": self.collection_size,
            "File Type": self.file_type.value.upper(),
            "Source": self.source_config.source.upper(),
        }

        names = self._dataset.schema.names
        cols = [c for c in ("Modality", "BodyPartExamined") if c in names]
        if not cols:
            summary["Modalities"] = []
            summary["BodyPartsExamined"] = []
            return summary

        narrow = self.read_index_rows(modalities=None, columns=cols)
        if "Modality" in narrow.columns:
            summary["Modalities"] = narrow["Modality"].dropna().unique().tolist()
        else:
            summary["Modalities"] = []
        if "BodyPartExamined" in narrow.columns:
            summary["BodyPartsExamined"] = (
                narrow["BodyPartExamined"].dropna().unique().tolist()
            )
        else:
            summary["BodyPartsExamined"] = []

        return summary
