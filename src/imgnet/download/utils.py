import s3fs
from pathlib import Path
from tqdm import tqdm
import requests
import shutil

from imgnet.loggers import logger


def download_file_from_s3(bucket_name: str, file_name: str, output_path: Path, chunk_size: int = 2**20) -> None:
    fs = s3fs.S3FileSystem(anon=True)
    file_path = f"{bucket_name}/{file_name}"
    out_file = output_path / Path(file_name).name

    # Get size for progress bar and ETA
    size = fs.info(file_path)["size"]

    with tqdm(total=size, unit="B", unit_scale=True, desc=file_name) as pbar:
        with fs.open(file_path, "rb") as remote:
            with open(out_file, "wb") as local:
                while True:
                    chunk = remote.read(chunk_size)
                    if not chunk:
                        break
                    local.write(chunk)
                    pbar.update(len(chunk))


def download_from_dropbox(url: str, output_path: Path, chunk_size: int = 2**20) -> Path:
    # Force direct download
    dl_url = url.replace("dl=0", "dl=1").replace("www.dropbox.com", "dl.dropboxusercontent.com")

    with requests.get(dl_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        filename = output_path / url.split("/")[-1].split("?")[0]

        with tqdm(total=total, unit="B", unit_scale=True, desc=filename.name) as pbar:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
    return filename


def download_from_zenodo(
    record_id: str,
    output_path: Path,
    filename: str | None = None,
    chunk_size: int = 2**20,
) -> list[Path]:
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}")
    resp.raise_for_status()
    files = resp.json()["files"]

    if filename is not None:
        files = [f for f in files if f["key"] == filename]
        if not files:
            available = [f["key"] for f in resp.json()["files"]]
            raise FileNotFoundError(
                f"File '{filename}' not found in Zenodo record {record_id}. "
                f"Available files: {available}"
            )

    downloaded = []
    for f in files:
        url = f["links"]["self"]
        size = f["size"]
        out_file = output_path / f["key"]

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tqdm(total=size, unit="B", unit_scale=True, desc=f["key"]) as pbar:
                with open(out_file, "wb") as out:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        out.write(chunk)
                        pbar.update(len(chunk))
        downloaded.append(out_file)
    return downloaded


def download_latest_release_asset(
    owner: str,
    repo: str,
    asset_name: str,
    download_dir: str | Path = "downloads",
    token: str | None = None,
) -> Path:
    """
    Download a specific asset from the latest GitHub release.

    Parameters
    ----------
    owner : str
        Repository owner
    repo : str
        Repository name
    asset_name : str
        Name of the asset to download
    download_dir : str | Path
        Directory to save the file
    token : str | None
        GitHub token (optional)
    """

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"

    release = requests.get(api_url, headers=headers).json()
    assets = release.get("assets", [])

    asset = next((a for a in assets if a["name"] == asset_name), None)

    if asset is None:
        raise ValueError(f"Asset '{asset_name}' not found in latest release. Available assets: {assets}")

    url = asset["browser_download_url"]

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    filepath = download_dir / asset_name

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(filepath, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=asset_name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    return filepath


def _post_unzip(output_path: Path, archive_filenames: list[str] | None = None) -> None:
    """Unpack every archive found in *output_path*."""
    for archive in output_path.iterdir():
        if archive_filenames is not None and archive.name not in archive_filenames:
            continue
        if archive.suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            logger.info(f"Unpacking {archive.name}")
            shutil.unpack_archive(archive, output_path)
            archive.unlink()