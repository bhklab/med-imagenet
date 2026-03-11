import pandas as pd
from imgnet.utils import get_idc_client


def _fetch_collection_description_tcia(collection: str) -> str:
    """Fetch the description of a collection from the TCIA index."""
    collection = collection.lower().replace(" ", "_").replace("-", "_")
    client = get_idc_client()
    path = client.indices_overview["collections_index"]["file_path"]
    collections_df = pd.read_parquet(path)
    description = collections_df.loc[collections_df["collection_id"] == collection, "Description"].iloc[0]
    return description