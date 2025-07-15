import asyncio
from pathlib import Path

import pandas as pd
from pydicom import dcmread
from imgtools.dicom.crawl import Crawler
from nbiatoolkit import NBIA_ENDPOINT
from nbiatoolkit.nbia import NBIAClient
from nbiatoolkit.dicomtags.tags import generateFileDatasetFromTags

def index_collection(client: NBIAClient, collection: str, output_path: Path) -> None:
    """
    Index a collection of DICOM files from the NBIA API using med-imagetools.

    Args:
        client: NBIAClient instance
        collection: str
        output_path: Path
    """

    series = client.getSeries({'Collection': collection})

    for s in series:
        # Need to download modalities where reference series is needed
        if s["Modality"] not in ["CT", "PT", "MR"]:
            sop_uid = client.getSOPIDs(s)

            for key in sop_uid:
                sop_uid = sop_uid[key][0]["SOPInstanceUID"]
            series_uid = s["SeriesInstanceUID"]
            file = asyncio.run(
                client.query_bytes(NBIA_ENDPOINT.DOWNLOAD_IMAGE.value, {"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid})
            )
            ds = dcmread(file, stop_before_pixels=True, force=True)
        else: 
            tags = asyncio.run(
                client.query_json(NBIA_ENDPOINT.GET_DICOM_TAGS.value, {"SeriesUID": s["SeriesInstanceUID"]})
            )
            tags_df = pd.DataFrame(tags)
            ds = generateFileDatasetFromTags(tags_df)

        ds.save_as(output_path / collection / f"{s['SeriesInstanceUID']}.dcm", enforce_file_format=False)


    crawler = Crawler(output_path / collection, force=True)
    crawler.crawl()

if __name__ == "__main__":
    client = NBIAClient()

    collections = client.getCollections()

    output_path = Path("indexed_datasets")
    output_path.mkdir(parents=True, exist_ok=True)

    for _collection in collections:
        collection = _collection["Collection"]
        index_collection(client, collection, output_path)