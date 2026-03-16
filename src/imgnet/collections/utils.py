import pandas as pd
from pathlib import Path
import os
from platformdirs import user_data_dir
from imgnet.utils import get_idc_client


_ENV_VAR = "IMGNET_INDEX_DIR"
_APP_NAME = "med-imagenet"
_APP_AUTHOR = "bhklab"


def _default_indexed_datasets_path() -> Path:
    """Return the default path for the ``indexed_datasets/`` directory.

    Resolution order:
    1. ``$IMGNET_INDEX_DIR/indexed_datasets`` if the env var is set.
    2. ``<platform-data-dir>/med-imagenet/indexed_datasets`` via *platformdirs*.
    """
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env) / "indexed_datasets"
    return Path(user_data_dir(_APP_NAME, _APP_AUTHOR)) / "indexed_datasets"


def _fetch_collection_description_tcia(collection: str) -> str:
    """Fetch the description of a collection from the TCIA index."""
    collection = collection.lower().replace(" ", "_").replace("-", "_")
    client = get_idc_client()
    path = client.indices_overview["collections_index"]["file_path"]
    collections_df = pd.read_parquet(path)
    description = collections_df.loc[collections_df["collection_id"] == collection, "Description"].iloc[0]
    return description

