from pathlib import Path

import pandas as pd
from idc_index import IDCClient

from imgnet.utils import get_idc_client
from imgnet.query import ValidQuery
from imgnet.collections.store import IndexedDatasets
from imgnet.download import download_collection
from imgnet.loggers import logger


class ImgNet:
    def __init__(
        self,
        output_path: Path,
        store: IndexedDatasets,
        client: IDCClient | None = None,
    ):
        self.output_path = Path(output_path)
        self.store = store
        self.client = client if client is not None else get_idc_client()

    def download(self, collection: str, series_uids: list[str] | None = None) -> None:
        """Download a collection (or specific series within it).

        Parameters
        ----------
        collection : str
            The collection name.
        series_uids : list[str] | None
            For DICOM/TCIA collections, the specific series to download.
            Ignored for non-TCIA sources which download the full collection.
        """
        download_collection(
            output_path=self.output_path,
            config=self.store.source_config(collection),
            client=self.client,
            series_uids=series_uids,
        )

    def query(self, valid_query: ValidQuery, download: bool = False) -> pd.DataFrame:
        """Query indexed datasets and optionally download selected series.

        Parameters
        ----------
        valid_query : ValidQuery
            The query used to select series from indexed collections.
        download : bool
            If true, downloads the selected series after querying.

        Returns
        -------
        pd.DataFrame
            DataFrame of matched series.
        """
        results = valid_query.process(self.store)

        if download:
            for collection, group in results.groupby("Collection"):
                series_uids = group["SeriesInstanceUID"].tolist()
                logger.info(f"Downloading {len(series_uids)} series from {collection}")
                self.download(collection, series_uids=series_uids)

        return results


if __name__ == "__main__":
    store = IndexedDatasets(Path.cwd() / "indexed_datasets")
    imgnet = ImgNet(output_path=Path("bruh"), store=store)

    imgnet.download(
        "4D-Lung",
        series_uids=["1.3.6.1.4.1.14519.5.2.1.6834.5010.124741849880980303405787216373"],
    )
