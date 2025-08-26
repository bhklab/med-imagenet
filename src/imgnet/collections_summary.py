from imgnet.supported_collections import SUPPORTED_COLLECTIONS

from tqdm import tqdm
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from imgtools.dicom import Interlacer
from pathlib import Path
import pandas as pd
import json
from rich.table import Table
from rich import print

class Collections():

    def display_collections(self, body_part: list[str], modality: list[str]) -> None:
        """Display all supported collections.
        
        Parameters
        ----------
        
        body_part: `list[str]`
            the list of body parts to look for in collections
        modality: `list[str]`
            the list of modalities to look for in collections
            
        Returns
        -------
        
        Prints a table of available collections and saves data to collections_summary.json."""
        table = Table(title="Collections Summary")
        table.add_column("Collection", justify="right")
        table.add_column("BodyPartsExamined", justify="left")
        table.add_column("Modalities", justify="left")
        table.add_column("Series Count", justify="right")
        table.add_column("Size", justify="right")
        json_path = Path(__file__).parent.parent.parent / "indexed_datasets/collections_summary.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                collection_db = json.load(f)
        else:
            sizes = self.get_collection_sizes()
            collection_db = {}
            for collection in SUPPORTED_COLLECTIONS:
                # get the indexed_datasets filepath
                file_path = Path(__file__).parent.parent.parent / "indexed_datasets/.imgtools" / collection / "crawl_db.json"
                
                with open(file_path, "r") as f:
                    crawl_json = json.load(f)
                
                collection_summary = {
                    "Modalities": set(),
                    "BodyPartsExamined": set(),
                    "SeriesCount": 0,
                    "Size": "".join(sizes[collection])
                }
                for key in crawl_json:
                    series = crawl_json[key][list(crawl_json[key].keys())[0]] # I know this looks bad
                    if series["Modality"]:
                        collection_summary["Modalities"].add(series["Modality"])
                    if series["BodyPartExamined"]:
                        collection_summary["BodyPartsExamined"].add(series["BodyPartExamined"])
                    collection_summary["SeriesCount"] += 1
                for key in collection_summary:
                    if isinstance(collection_summary[key], set):
                        collection_summary[key] = list(collection_summary[key])
                collection_db[collection] = collection_summary
        
        for collection in collection_db:
            collection_summary = collection_db[collection]
            contains_body_part = True
            contains_modality = True
            if body_part and not any(part in collection_summary["BodyPartsExamined"] for part in body_part):
                contains_body_part = False
            if modality and not any(m in collection_summary["Modalities"] for m in modality):
                contains_modality = False
            if contains_body_part and contains_modality:
                table.add_row(collection, ", ".join(collection_summary["BodyPartsExamined"]), ", ".join(collection_summary["Modalities"]), f"{collection_summary['SeriesCount']}", collection_summary["Size"])
        print(table)
        with open(json_path, "w") as f:
            json.dump(collection_db, f)


    def fetch_collection_size(self, _collection):
        collection = _collection.replace(" ", "-")
        url = f"https://www.cancerimagingarchive.net/collection/{collection}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            r = requests.get(url, headers=headers, timeout=10)
            text = r.text.replace("\n", "")
            if r.ok:
                pattern = r'(?:Download\(|<nobr>)([\d.]+)\s*(KB|MB|GB|TB)(?:\)|</nobr>)'
                matches = re.findall(pattern, text)
                if matches:
                    return _collection, f"{matches[0][0]} {matches[0][1]}"  # first match only
                else:
                    return _collection, "N/A"
            else:
                return _collection, "N/A"
        except Exception as e:
            return _collection, "N/A"

    def get_collection_sizes(self):
        sizes = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.fetch_collection_size, c): c for c in SUPPORTED_COLLECTIONS}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Scraping collections"):
                collection, size = f.result()
                sizes[collection] = size
        return sizes


