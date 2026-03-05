import requests
import re


def _fetch_collection_size(name: str) -> tuple[str, str]:
    collection = name.replace(" ", "-")
    url = f"https://www.cancerimagingarchive.net/collection/{collection}/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        text = r.text.replace("\n", "")
        if r.ok:
            pattern = r'(?:Download\(|<nobr>)([\d.]+)\s*(KB|MB|GB|TB)(?:\)|</nobr>)'
            matches = re.findall(pattern, text)
            if matches:
                return name, f"{matches[0][0]} {matches[0][1]}"
        return name, "N/A"
    except Exception:
        return name, "N/A"