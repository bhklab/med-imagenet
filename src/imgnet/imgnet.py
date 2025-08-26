from pathlib import Path

from pydicom import dcmread

from concurrent.futures import ThreadPoolExecutor, as_completed

from nbiatoolkit.nbia import NBIAClient
from nbiatoolkit import NBIA_ENDPOINT
import asyncio

from imgnet.query import ValidQuery

from imgnet.loggers import logger
import pandas as pd

class ImgNet:
    def __init__(self, output_path: Path, client: NBIAClient):
        self.output_path = Path(output_path)
        self.client = client

    def download_image(self, series_uid: str, max_workers: int = 8) -> None:
        """
        Download a series using NBIA Toolkit and save in `self.output_path`.

        Parameters
        ----------
        series_uid: `str`
            The SeriesUID of the series to be downloaded.
        max_workers: `int`
            Maximum number of worker threads for parallel downloading.

        Returns
        -------
        `None`
        """
        logger.info(f"Downloading image for series {series_uid}")

        sop_uid_data = self.client.getSOPIDs({"SeriesInstanceUID": series_uid})

        save_path = Path(self.output_path / f"{series_uid}")
        save_path.mkdir(exist_ok=True, parents=True)

        def download_and_save(sop):
            sop_uid = sop["SOPInstanceUID"]
            dicom_bytes = self.client.download_single_image(series_uid, sop_uid)
            ds = dcmread(dicom_bytes, force=True)
            ds.save_as(save_path / f"{sop_uid}.dcm")
            return sop_uid

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key in sop_uid_data:
                for sop in sop_uid_data[key]:
                    futures.append(executor.submit(download_and_save, sop))

            for f in as_completed(futures):
                try:
                    sop_uid = f.result()
                    logger.info(f"Downloaded SOPInstanceUID {sop_uid}")
                except Exception as e:
                    logger.error(f"Error downloading SOP: {e}")


    def query(self, valid_query: ValidQuery, download: bool = False) -> pd.DataFrame:
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
            series_results = results["SeriesInstanceUID"].tolist()
            for series in series_results:
                self.download_image(series)
            
        

        return results



