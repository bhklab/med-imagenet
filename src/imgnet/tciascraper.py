from .supported_collections import SUPPORTED_COLLECTIONS

from tqdm import tqdm
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def fetch_collection_size(_collection):
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

def get_collection_sizes():
    sizes = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_collection_size, c): c for c in SUPPORTED_COLLECTIONS}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Scraping collections"):
            collection, size = f.result()
            sizes[collection] = size
    return sizes


