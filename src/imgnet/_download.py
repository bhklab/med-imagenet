import s3fs
from pathlib import Path
from tqdm import tqdm
import requests

# ------------------------------------------------------------
# Download Helper Functions
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Download Datasets
# ------------------------------------------------------------

def download_medical_decathlon(output_path: Path) -> None:
    tasks = [
        'Task01_BrainTumour.tar',
        'Task02_Heart.tar',
        'Task03_Liver.tar',
        'Task04_Hippocampus.tar',
        'Task05_Prostate.tar',
        'Task06_Lesion.tar',
        'Task07_Pancreas.tar',
        'Task08_HepaticVessel.tar',
        'Task09_Spleen.tar',
        'Task10_Colon.tar'
    ]
    for task in tasks:
        download_file_from_s3('msd-for-monai', task, output_path)


def download_total_segmentator_data(output_path: Path) -> None:
    url = "https://www.dropbox.com/scl/fi/oq0fsz8oauory204g8o6f/Totalsegmentator_dataset_v201.zip?rlkey=afnl2ixhqca2ukkf1v9p6jz7p&dl=0"
    download_from_dropbox(url, output_path)

def download_total_segmentator_mri_data(output_path: Path) -> None:
    url = "https://www.dropbox.com/scl/fi/cn7736fcznvquf33m7yod/TotalsegmentatorMRI_dataset_v100.zip?rlkey=af1l0x7ajnczrminfzzb6ict9&st=if883tqe&dl=0"
    download_from_dropbox(url, output_path)



if __name__ == "__main__":
    output_path = Path("temp_data")
    output_path.mkdir(exist_ok=True, parents=True)
    # download_medical_decathlon(output_path)
    # download_total_segmentator_data(output_path)
