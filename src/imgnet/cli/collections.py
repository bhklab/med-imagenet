import click
from imgtools.dicom import Interlacer
from pathlib import Path
import pandas as pd
import json
from rich.table import Table
from rich import print
from imgnet.tciascraper import get_collection_sizes

from imgnet.supported_collections import SUPPORTED_COLLECTIONS
@click.command(no_args_is_help=False)
@click.help_option(
    "--help",
    "-h"
)
def collections():
    """Display all supported collections."""
    table = Table(title="Collections Summary")
    table.add_column("Collection", justify="right")
    table.add_column("BodyPartsExamined", justify="left")
    table.add_column("Modalities", justify="left")
    table.add_column("Series Count", justify="right")
    table.add_column("Size", justify="right")


    sizes = get_collection_sizes()
    collection_db = {}
    for collection in SUPPORTED_COLLECTIONS:
        # get the indexed_datasets filepath
        
        file_path = Path(__file__).parent.parent.parent.parent / "indexed_datasets/.imgtools" / collection / "crawl_db.json"
        
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
        table.add_row(collection, ", ".join(collection_summary["BodyPartsExamined"]), ", ".join(collection_summary["Modalities"]), f"{collection_summary["SeriesCount"]}", collection_summary["Size"])
    print(table)
    with open(Path(__file__).parent.parent.parent.parent / "indexed_datasets/collections_summary.json", "w") as f:
        json.dump(collection_db, f)

    
            


