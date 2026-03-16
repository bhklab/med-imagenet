import shutil
from pathlib import Path

import requests
import s3fs
from tqdm import tqdm

from imgnet.loggers import logger
from imgnet.utils import get_idc_client


def list_s3_bucket_keys(bucket_name: str) -> list[str]:
    """List all object keys in an S3 bucket (anonymous). Skips directory markers."""
    fs = s3fs.S3FileSystem(anon=True)
    prefix = f"{bucket_name}/"
    paths = fs.find(bucket_name)
    return [p[len(prefix) :] for p in paths if not p.endswith("/")]


def download_file_from_s3(
    bucket_name: str,
    file_name: str,
    output_path: Path,
    chunk_size: int = 2**20,
    dry_run: bool = False,
) -> float | None:
    """Download one file from S3. If dry_run=True, return size in bytes and do not download."""
    fs = s3fs.S3FileSystem(anon=True)
    file_path = f"{bucket_name}/{file_name}"
    size = fs.info(file_path)["size"]
    if dry_run:
        return float(size)
    out_file = output_path / Path(file_name).name
    with tqdm(total=size, unit="B", unit_scale=True, desc=file_name) as pbar:  # noqa: SIM117
        with fs.open(file_path, "rb") as remote:
            with out_file.open("wb") as local:
                while True:
                    chunk = remote.read(chunk_size)
                    if not chunk:
                        break
                    local.write(chunk)
                    pbar.update(len(chunk))
    return None


def download_from_dropbox(
    url: str,
    output_path: Path,
    chunk_size: int = 2**20,
    dry_run: bool = False,
) -> Path | float:
    """Download from Dropbox URL. If dry_run=True, return size in bytes and do not download."""
    dl_url = url.replace("dl=0", "dl=1").replace(
        "www.dropbox.com", "dl.dropboxusercontent.com"
    )

    with requests.get(dl_url, stream=True) as r:
        r.raise_for_status()
        total = r.headers.get("content-length", 0)

        if dry_run:
            return float(total)

        filename = output_path / url.split("/")[-1].split("?")[0]
        with tqdm(
            total=int(total), unit="B", unit_scale=True, desc=filename.name
        ) as pbar:  # noqa: SIM117
            with filename.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
    return filename


def download_from_zenodo(
    record_id: str,
    output_path: Path,
    filename: str | None = None,
    filenames: list[str] | None = None,
    chunk_size: int = 2**20,
    dry_run: bool = False,
) -> list[Path] | float:
    """Download files from a Zenodo record. If dry_run=True, return total size in bytes and do not download."""
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}")
    resp.raise_for_status()
    files = resp.json()["files"]

    if filename is not None:
        files = [f for f in files if f["key"] == filename]
        if not files:
            available = [f["key"] for f in resp.json()["files"]]
            msg = (
                f"File '{filename}' not found in Zenodo record {record_id}. "
                f"Available files: {available}"
            )
            raise FileNotFoundError(msg)
    elif filenames is not None:
        keys = set(filenames)
        files = [f for f in files if f["key"] in keys]

    if dry_run:
        return float(sum(f["size"] for f in files))

    downloaded = []
    for f in files:
        url = f["links"]["self"]
        size = f["size"]
        out_file = output_path / f["key"]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tqdm(
                total=size, unit="B", unit_scale=True, desc=f["key"]
            ) as pbar:  # noqa: SIM117
                with out_file.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        out.write(chunk)
                        pbar.update(len(chunk))
        downloaded.append(out_file)
    return downloaded


def _fetch_collection_size_idc(collection_id: str) -> float:
    """Fetch the size of a collection from the IDC index in GB."""
    collection_id = collection_id.lower().replace(" ", "_").replace("-", "_")
    client = get_idc_client()
    size = client.collection_summary.loc[collection_id, "series_size_MB"]

    return round(float(size / 1000), 2)


def _post_unzip(
    output_path: Path, archive_filenames: list[str] | None = None
) -> None:
    """Unpack every archive found in *output_path*."""
    for archive in output_path.iterdir():
        if (
            archive_filenames is not None
            and archive.name not in archive_filenames
        ):
            continue
        if archive.suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            logger.info(f"Unpacking {archive.name}")
            shutil.unpack_archive(archive, output_path)
            archive.unlink()
