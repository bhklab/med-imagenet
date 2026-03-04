from pathlib import Path
import pandas as pd
from idc_index import IDCClient

from imgnet.query import ValidQuery
from imgnet.loggers import logger, tqdm_logging_redirect


class ImgNet:
    def __init__(
        self, 
        output_path: Path, 
        client: IDCClient,
        indexed_datasets_path: Path | str | None = None
    ):
        self.output_path = Path(output_path)
        self.client = client

        if indexed_datasets_path is None:
            indexed_datasets_path = Path.cwd() / "indexed_datasets"
        self.indexed_datasets_path = Path(indexed_datasets_path)


    def download_image(self, series_uid: str) -> None:
        """
        Download a series using idc-index and save in `self.output_path`.

        Parameters
        ----------
        series_uid: `str`
            The SeriesUID of the series to be downloaded.

        Returns
        -------
        `None`
        """
        logger.info(f"Downloading image for series {series_uid}")

        save_path = Path(self.output_path / f"{series_uid}")
        save_path.mkdir(exist_ok=True, parents=True)

        with tqdm_logging_redirect():
            self.client.download_dicom_series(series_uid, save_path)


    def query(self, valid_query: ValidQuery, download: bool = False) -> pd.DataFrame:
        """
        Query crawled TCIA datasets and optionally download selected DICOMs using idc-index.

        Parameters
        ----------

        valid_query: `ValidQuery`
            The query used to select seriesUIDs from TCIA.
        download: `bool`
            If true, downloads the selected series' using idc-index.

        Returns
        -------
        `dict[str: list[str]]`
            A dictionary containing a list of selected seriesUIDs for each collection in the query.
        """
        results = valid_query.process(root_dir=self.indexed_datasets_path)
        
        if download:
            series_results = results["SeriesInstanceUID"].tolist()
            for series in series_results:
                self.download_image(series)
            
        return results



if __name__ == "__main__":
    client = IDCClient()
    imgnet = ImgNet(output_path=Path("bruh"), client=client)

    imgnet.download_image("1.3.6.1.4.1.14519.5.2.1.6834.5010.124741849880980303405787216373")