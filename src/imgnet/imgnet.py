from pathlib import Path

from pydicom import dcmread
import pandas as pd
from nbiatoolkit.nbia import NBIAClient
from nbiatoolkit import NBIA_ENDPOINT

from imgnet.query import ValidQuery
import json

from imgnet.loggers import logger

class ImgNet:
    def __init__(self, output_path: Path, client: NBIAClient):
        self.output_path = output_path
        self.client = client

    def download_image(self, series_uid: str) -> None:
        """
        Download a series using NBIA Toolkit and save in `self.output_path`.

        Parameters
        ----------

        series_uid: `str`
            The SeriesUID of the series to be downloaded.

        Returns
        -------
        `None`
        """
        logger.info(f"Downloading image for series {series_uid}")
        series_bytes = self.client.download_series(series_uid)

        ds = dcmread(series_bytes, force=True)
        ds.save_as(self.output_path / f"{series_uid}.dcm")


    def query(self, valid_query: ValidQuery, download: bool = False) -> dict[str: list[str]]:
        """
        Query crawled TCIA datasets and optionally download selected DICOMs using NBIA Toolkit.

        Parameters
        ----------

        valid_query: `ValidQuery`
            The query used to select seriesUIDs from TCIA.
        download: `bool`
            If true, downloads the selected series' using NBIA Toolkit.

        Returns
        -------
        `dict[str: list[str]]`
            A dictionary containing a list of selected seriesUIDs for each collection in the query.
        """
        results = valid_query.process()
        
        if download:
            for key in results:
                for series in results[key]:
                    self.download_image(series)
            
        

        return results



