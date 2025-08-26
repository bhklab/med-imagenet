import asyncio
from typing import Literal
from pathlib import Path
import concurrent.futures
from functools import partial

from tqdm import tqdm
import pandas as pd
from pydicom import dcmread
from imgtools.dicom.crawl import Crawler
from nbiatoolkit import NBIA_ENDPOINT
from nbiatoolkit.nbia import NBIAClient
from nbiatoolkit.dicomtags.tags import generateFileDatasetFromTags

from loggers import logger


def process_single_series(client: NBIAClient, s: dict, output_path: Path, collection: str, exist_strategy: Literal["skip", "overwrite"] = "skip") -> bool:
    """
    Process a single series. Returns True if successful, False otherwise.
    """
    if exist_strategy == "skip" and (output_path / collection / "images" / f"{s['SeriesInstanceUID']}.dcm").exists():
        return True
    
    try:
        if s["Modality"] not in ["CT", "PT", "MR"]:
            sop_uid_data = client.getSOPIDs(s)

            for key in sop_uid_data:
                sop_uid = sop_uid_data[key][0]["SOPInstanceUID"]
            series_uid = s["SeriesInstanceUID"]
            
            file = asyncio.run(
                client.query_bytes(NBIA_ENDPOINT.DOWNLOAD_IMAGE, {"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid})
            )
            ds = dcmread(file, stop_before_pixels=True, force=True)
        else: 
            tags = asyncio.run(
                client.query_json(NBIA_ENDPOINT.GET_DICOM_TAGS, {"SeriesUID": s["SeriesInstanceUID"]})
            )
            tags_df = pd.DataFrame(tags)
            ds = generateFileDatasetFromTags(tags_df)

        ds.save_as(output_path / collection / "images" / f"{s['SeriesInstanceUID']}.dcm", enforce_file_format=False)
        return True
        
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Failed to process series {s['SeriesInstanceUID']}: {e}")
        return False


def index_collection(client: NBIAClient, collection: str, output_path: Path, exist_strategy: Literal["skip", "overwrite"] = "skip", max_workers: int = 1) -> None:
    """
    Index a collection of DICOM files from the NBIA API using med-imagetools.

    Args:
        client: NBIAClient instance
        collection: str
        output_path: Path
        exist_strategy: Strategy for handling existing files
        max_workers: Number of parallel workers (1 = sequential, >1 = parallel)
    """
    series = client.getSeries({'Collection': collection})
    print(series)
    # logger.info(f"Indexing collection {collection}, {len(series)} series found")

    # (output_path / collection / "images").mkdir(parents=True, exist_ok=True)
    
    # # Filter out existing files if skip strategy is used
    # if exist_strategy == "skip":
    #     series = [s for s in series if not (output_path / collection / "images" / f"{s['SeriesInstanceUID']}.dcm").exists()]
    #     logger.info(f"After filtering existing files: {len(series)} series to process")
    
    # if not series:
    #     logger.info(f"No new series to process for collection {collection}")
    #     return
    
    # # Create a partial function with fixed arguments
    # process_func = partial(process_single_series, client, output_path=output_path, collection=collection, exist_strategy=exist_strategy)
    
    # if max_workers == 1:
    #     # Sequential processing (original behavior)
    #     with tqdm(
    #         series, 
    #         desc=f"Processing {collection}",
    #         unit="series",
    #         total=len(series),
    #         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    #     ) as pbar:
    #         for s in pbar:
    #             result = process_func(s)
    #             pbar.set_postfix({"Success": result})
    # else:
    #     # Parallel processing
    #     logger.info(f"Processing {len(series)} series with {max_workers} workers")
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         # Submit all tasks
    #         future_to_series = {executor.submit(process_func, s): s for s in series}
            
    #         # Process results as they complete
    #         successful = 0
    #         failed = 0
            
    #         with tqdm(
    #             total=len(series),
    #             desc=f"Processing {collection}",
    #             unit="series",
    #             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    #         ) as pbar:
    #             for future in concurrent.futures.as_completed(future_to_series):
    #                 result = future.result()
    #                 if result:
    #                     successful += 1
    #                 else:
    #                     failed += 1
                    
    #                 pbar.update(1)
    #                 pbar.set_postfix({"Success": successful, "Failed": failed})
        
    #     logger.info(f"Collection {collection} processing complete: {successful} successful, {failed} failed")

    # logger.info(f"Crawling collection {collection}")
    # crawler = Crawler(output_path / collection, force=True)
    # crawler.crawl()

    # logger.info(f"Finished indexing collection {collection}, output path: {output_path / collection}")

if __name__ == "__main__":
    client = NBIAClient()

    collections = client.getCollections()

    output_path = Path("indexed_datasets")
    output_path.mkdir(parents=True, exist_ok=True)

    for _collection in collections:
        collection = _collection["Collection"]

        if (output_path / ".imgtools" / collection / "index.csv").exists():
            logger.info(f"Skipping collection {collection} because it already exists")
            continue

        index_collection(client, collection, output_path, exist_strategy="overwrite", max_workers=int(client.max_concurrent_requests) - 3)