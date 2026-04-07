import time
from pathlib import Path

import pandas as pd
from idc_index import IDCClient

from imgnet.collections.store import IndexedDatasets
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
            collection_str = str(collection)
            if self.store.source_config(collection_str).source == "tcia":
                instance_ids = group["SeriesInstanceUID"].tolist()
            else:
                instance_ids = group["filepath"].tolist()
            logger.info(
                f"Downloading {collection_str} {len(instance_ids)} instances from {self.store.source_config(collection_str).source.capitalize()}"
            )
            time_start = time.time()
            self.store.downloader(collection_str).download(
                output_path=self.output_path / collection_str,
                instance_ids=instance_ids,
            )
            time_end = time.time()
            logger.info(
                f"Download completed in {time_end - time_start:.2f} seconds"
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
