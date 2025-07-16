from pathlib import Path

from pydicom import dcmread
import pandas as pd
from nbiatoolkit.nbia import NBIAClient
from nbiatoolkit import NBIA_ENDPOINT

from loggers import logger

class ImgNet:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.client = NBIAClient()

    @property
    def database_path(self) -> Path:
        path = Path("indexed_datasets")

        # if not path.exists():
        #     requests.get("{PATH TO DB}", stream=True)

        return path

    def download_image(self, series_uid: str) -> None:
        logger.info(f"Downloading image for series {series_uid}")
        series_bytes = self.client.download_series(series_uid)

        ds = dcmread(series_bytes, force=True)
        ds.save_as(self.output_path / f"{series_uid}.dcm")


    def query(self, query: str, collection: str, download: bool = False) -> pd.DataFrame:
        logger.info(f"Querying for {query} in {collection}")
        df = pd.read_csv(self.database_path / collection, sep="\t")
        df = df.query(query)

        if download:
            for _, row in df.iterrows():
                self.download_image(row["SeriesInstanceUID"])
                
        return df



