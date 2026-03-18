from pathlib import Path

import pandas as pd
from idc_index import IDCClient

from imgnet.collections.store import IndexedDatasets
from imgnet.download import download_collection
from imgnet.loggers import logger
from imgnet.query import ValidQuery
from imgnet.utils import get_idc_client


class ImgNet:
    def __init__(
        self,
        output_path: Path,
        store: IndexedDatasets,
        client: IDCClient | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.store = store
        self.client = client if client is not None else get_idc_client()

    def download(self, results: pd.DataFrame) -> None:
        """Download the series from the query results.

        Parameters
        ----------
        results : pd.DataFrame
            The results of the query.
        """

        for collection, group in results.groupby("Collection"):
            collection = str(collection)
            if self.store.source_config(collection).source == "tcia":
                instance_ids = group["SeriesInstanceUID"].tolist()
            else:
                instance_ids = group["filepath"].tolist()
            logger.info(
                f"Downloading {collection} {len(instance_ids)} instances from {self.store.source_config(collection).source.capitalize()}"
            )
            download_collection(
                output_path=self.output_path / collection,
                config=self.store.source_config(collection),
                instance_ids=instance_ids,
            )

    def query(self, valid_query: ValidQuery) -> pd.DataFrame:
        """Query indexed datasets and optionally download selected series.

        Parameters
        ----------
        valid_query : ValidQuery
            The query used to select series from indexed collections.

        Returns
        -------
        pd.DataFrame
            DataFrame of matched series.
        """
        results = valid_query.process(self.store)

        return results
