import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import s3fs
from tqdm import tqdm
from tqdm.auto import tqdm as _tqdm

from imgnet.download.base import BaseDownloader
from imgnet.download.utils import _download_http_file
from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.utils import RemoteArchive, get_idc_client, get_nbia_client


os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_url, snapshot_download


class HuggingFaceDownloader(BaseDownloader):
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from Hugging Face. Here instance_ids is a list of filenames to download."""

        if instance_ids is None:
            logger.info(
                f"Downloading all instances from Hugging Face repository {self.repo_id}"
            )
            files_to_download = self.members
        else:
            logger.info(
                f"Downloading {len(instance_ids)} instances from Hugging Face repository {self.repo_id}"
            )
            remaining = set(instance_ids)
            files_to_download = []

            for file_name in self.members:
                if file_name in remaining:
                    files_to_download.append(file_name)
                    remaining.remove(file_name)
                    continue

                if RemoteArchive.is_supported_archive(file_name) and remaining:
                    archive_url = hf_hub_url(
                        repo_id=self.repo_id,
                        filename=file_name,
                        repo_type="dataset",
                    )
                    archive = RemoteArchive(
                        archive_url, Path(file_name).suffix
                    )
                    extracted = archive.extract(
                        filenames=sorted(remaining),
                        output_path=output_path,
                    )
                    remaining -= set(extracted)

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in Hugging Face repository {self.repo_id}"
                logger.warning(msg)

        if len(files_to_download) > 0:
            with tqdm_logging_redirect():                    
                snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=output_path,
                    tqdm_class=_tqdm,
                    allow_patterns=files_to_download,
                    repo_type="dataset",
                    **kwargs,
                )

    @property
    def size(self) -> float:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(self.repo_id)
        used_storage = info.used_storage
        if used_storage is None:
            return 0.0
        return round(float(used_storage) / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(self.repo_id)
        siblings = info.siblings or []
        return [sibling.rfilename for sibling in siblings]


class ZenodoDownloader(BaseDownloader):
    def __init__(self, record_id: str) -> None:
        self.record_id = record_id

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download files from Zenodo. Here instance_ids is a list of filenames to download."""
        output_path.mkdir(parents=True, exist_ok=True)
        files = self.files_info

        if instance_ids is None:
            files_to_download = files
            logger.info(
                f"Downloading all files from Zenodo record {self.record_id}"
            )
        else:
            logger.info(
                f"Downloading {len(instance_ids)} files from Zenodo record {self.record_id}"
            )
            remaining = set(instance_ids)
            files_to_download = []

            for file_info in files:
                file_name = file_info["key"]
                if file_name in remaining:
                    files_to_download.append(file_info)
                    remaining.remove(file_name)
                    continue

                if RemoteArchive.is_supported_archive(file_name) and remaining:
                    archive = RemoteArchive(
                        file_info["links"]["self"], Path(file_name).suffix
                    )
                    extracted = archive.extract(
                        filenames=sorted(remaining),
                        output_path=output_path,
                    )
                    remaining -= set(extracted)

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in Zenodo record {self.record_id}"
                logger.warning(msg)

        for file_info in files_to_download:
            _download_http_file(
                url=file_info["links"]["self"],
                out_file=output_path / file_info["key"],
                desc=file_info["key"],
                size=file_info["size"],
            )

    @property
    def files_info(self) -> list[dict]:
        resp = requests.get(f"https://zenodo.org/api/records/{self.record_id}")
        resp.raise_for_status()
        if len(resp.json()["files"]) == 0:
            msg = f"No files found for Zenodo record {self.record_id}"
            raise FileNotFoundError(msg)

        return resp.json()["files"]

    @property
    def size(self) -> float:
        size = float(sum(f["size"] for f in self.files_info))
        return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        updated_file_names = []

        for file_info in self.files_info:
            file_name = file_info["key"]
            if RemoteArchive.is_supported_archive(file_name):
                archive = RemoteArchive(
                    file_info["links"]["self"], Path(file_name).suffix
                )
                updated_file_names.extend(archive.members)
            else:
                updated_file_names.append(file_name)

        return list(set(updated_file_names))


class DropboxDownloader(BaseDownloader):
    def __init__(self, url: str) -> None:
        self.url = url.replace("dl=0", "dl=1").replace(
            "www.dropbox.com", "dl.dropboxusercontent.com"
        )

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from Dropbox. Supports selecting specific instance_ids from archives."""
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = Path(urlparse(self.url).path).name

        if instance_ids is None:
            logger.info(
                f"Downloading all files from Dropbox source {self.url}"
            )
            _download_http_file(
                url=self.url,
                out_file=output_path / file_name,
                desc=file_name,
            )
            return None

        remaining = set(instance_ids)
        logger.info(
            f"Downloading {len(remaining)} files from Dropbox source {self.url}"
        )
        if file_name in remaining:
            _download_http_file(
                url=self.url,
                out_file=output_path / file_name,
                desc=file_name,
            )
            remaining.remove(file_name)

        if RemoteArchive.is_supported_archive(file_name) and remaining:
            archive = RemoteArchive(self.url, Path(file_name).suffix)
            extracted = archive.extract(
                filenames=sorted(remaining), output_path=output_path
            )
            remaining -= set(extracted)

        if remaining:
            msg = (
                f"Instance IDs {sorted(remaining)} not found in Dropbox source"
            )
            logger.warning(msg)

    @property
    def size(self) -> float:
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            size = float(r.headers.get("content-length", 0))
            return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        file_name = Path(urlparse(self.url).path)
        if RemoteArchive.is_supported_archive(str(file_name)):
            return RemoteArchive(self.url, file_name.suffix).members

        return [file_name.name]


class S3Downloader(BaseDownloader):
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self.fs = s3fs.S3FileSystem(anon=True)

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        all_bucket_files = self.fs.find(self.bucket_name)
        files_to_download = all_bucket_files

        if instance_ids is not None:
            remaining = set(instance_ids)
            files_to_download = []

            for file_path in all_bucket_files:
                file_name = Path(file_path).name
                if file_path in remaining or file_name in remaining:
                    files_to_download.append(file_path)
                    remaining.discard(file_path)
                    remaining.discard(file_name)
                    continue

                if RemoteArchive.is_supported_archive(file_path) and remaining:
                    archive = RemoteArchive(
                        f"s3://{file_path}", Path(file_path).suffix
                    )
                    extracted = archive.extract(
                        filenames=sorted(remaining), output_path=output_path
                    )
                    remaining -= set(extracted)

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in S3 bucket {self.bucket_name}"
                logger.warning(msg)

        for file_path in files_to_download:
            size = self.fs.info(file_path)["size"]
            out_file = output_path / Path(file_path).name

            with tqdm(  # noqa: SIM117
                total=size, unit="B", unit_scale=True, desc=out_file.name
            ) as pbar:
                with self.fs.open(file_path, "rb") as remote:
                    with out_file.open("wb") as local:
                        while True:
                            chunk = remote.read(2**20)
                            if not chunk:
                                break
                            local.write(chunk)
                            pbar.update(len(chunk))

    @property
    def size(self) -> float:
        size = float(
            sum(
                self.fs.info(file_path)["size"]
                for file_path in self.fs.find(self.bucket_name)
            )
        )
        return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        file_paths = [
            file_path for file_path in self.fs.find(self.bucket_name)
        ]
        expanded_members = set(file_paths)

        for file_path in file_paths:
            if RemoteArchive.is_supported_archive(file_path):
                archive = RemoteArchive(
                    f"s3://{file_path}", Path(file_path).suffix
                )
                expanded_members.update(archive.members)

        return list(expanded_members)


class IDCDownloader(BaseDownloader):
    def __init__(self, collection_id: str) -> None:
        self.collection_id = (
            collection_id.lower().replace(" ", "_").replace("-", "_")
        )
        self.client = get_idc_client()

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from IDC. Here instance_ids is a list of series UIDs to download."""

        if instance_ids is not None:
            if not all(
                instance_id in self.members for instance_id in instance_ids
            ):
                msg = f"Instance IDs {instance_ids} not found in IDC collection {self.collection_id}"
                raise ValueError(msg)
            series_uids = instance_ids
        else:
            logger.warning(
                f"No instance IDs provided, downloading all series from IDC collection {self.collection_id}"
            )
            series_uids = self.members

        output_path.mkdir(parents=True, exist_ok=True)
        with tqdm_logging_redirect():
            self.client.download_dicom_series(
                series_uids,
                output_path,
                dirTemplate="%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID",
            )

    @property
    def size(self) -> float:
        size = float(
            self.client.collection_summary.loc[
                self.collection_id, "series_size_MB"
            ]
        )
        return round(size / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        query = f"""
        SELECT
            collection_id,
            SeriesInstanceUID
        from index
        where collection_id = '{self.collection_id}'
        """
        return cast(
            "list[str]",
            self.client.sql_query(query)["SeriesInstanceUID"].tolist(),
        )
    
class NBIADownloader(BaseDownloader):
    def __init__(self, collection_id: str) -> None:
        self.collection_id = collection_id
        self.client = get_nbia_client()
        self._members = [item['SeriesInstanceUID'] for item in self.client.getSeries(self.collection_id)]


    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from NBIA. Here instance_ids is a list of series UIDs to download."""

        if instance_ids is not None:
            if not all(
                instance_id in self.members for instance_id in instance_ids
            ):
                msg = f"Instance IDs {instance_ids} not found in private TCIA collection {self.collection_id}"
                raise ValueError(msg)
            series_uids = instance_ids
        else:
            logger.warning(
                f"No instance IDs provided, downloading all series from private TCIA collection {self.collection_id}"
            )
            series_uids = self.members
        series_uids = [{"SeriesInstanceUID": uid} for uid in series_uids]
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with tqdm_logging_redirect():
            self.client.downloadSeries(
                series_uids,
                output_path
            )

    @property
    def size(self) -> float:
        return round(sum([item['FileSize'] for item in self.client.getSeries(self.collection_id)]) / 1000 ** 3, 2) # convert to GB

    @property
    def members(self) -> list[str]:
        return self._members


class GitHubDownloader(BaseDownloader):
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        # Parse owner/repo from repo_id
        parts = repo_id.split("/")
        if len(parts) != 2:
            msg = f"Invalid GitHub repo_id format: {repo_id}. Expected 'owner/repo'"
            raise ValueError(msg)
        self.owner, self.repo = parts
        self.api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}"

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from GitHub. Here instance_ids is a list of filenames to download."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        if instance_ids is None:
            logger.info(
                f"Downloading all instances from GitHub repository {self.repo_id}"
            )
            files_to_download = self.members
        else:
            logger.info(
                f"Downloading {len(instance_ids)} instances from GitHub repository {self.repo_id}"
            )
            remaining = set(instance_ids)
            files_to_download = []

            # Get all files from the GitHub repository
            all_files = self._get_repo_files()
            
            for file_info in all_files:
                file_name = file_info["name"]
                if file_name in remaining:
                    files_to_download.append(file_info)
                    remaining.remove(file_name)
                    continue

                if RemoteArchive.is_supported_archive(file_name) and remaining:
                    # For GitHub, we need to get the raw URL for the archive
                    archive_url = f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/main/{file_name}"
                    archive = RemoteArchive(
                        archive_url, Path(file_name).suffix
                    )
                    extracted = archive.extract(
                        filenames=sorted(remaining),
                        output_path=output_path,
                    )
                    remaining -= set(extracted)

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in GitHub repository {self.repo_id}"
                logger.warning(msg)

        for file_info in files_to_download:
            # Get the raw download URL
            download_url = file_info.get("download_url")
            if download_url:
                _download_http_file(
                    url=download_url,
                    out_file=output_path / file_info["name"],
                    desc=file_info["name"],
                    size=file_info.get("size", None),
                )

    def _get_repo_files(self) -> list[dict]:
        """Get all files from the GitHub repository."""
        # Get contents of the repository
        response = requests.get(f"{self.api_url}/contents")
        response.raise_for_status()
        contents = response.json()
        
        files = []
        for item in contents:
            if item["type"] == "file":
                files.append(item)
            elif item["type"] == "dir":
                # Recursively get files from subdirectories
                subdir_response = requests.get(item["url"])
                subdir_response.raise_for_status()
                subdir_contents = subdir_response.json()
                for subitem in subdir_contents:
                    if subitem["type"] == "file":
                        files.append(subitem)
        
        return files

    @property
    def size(self) -> float:
        """Get total size of repository in GB."""
        response = requests.get(f"{self.api_url}")
        response.raise_for_status()
        repo_info = response.json()
        size_kb = repo_info.get("size", 0)  # GitHub API returns size in KB
        return round(size_kb / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        """Get all filenames in the repository."""
        files = self._get_repo_files()
        updated_file_names = []

        for file_info in files:
            file_name = file_info["name"]
            if RemoteArchive.is_supported_archive(file_name):
                archive_url = f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/main/{file_name}"
                archive = RemoteArchive(
                    archive_url, Path(file_name).suffix
                )
                updated_file_names.extend(archive.members)
            else:
                updated_file_names.append(file_name)

        return list(set(updated_file_names))

class GoogleDriveDownloader(BaseDownloader):
    def __init__(self, file_id: str) -> None:
        self.file_id = file_id
        # For direct download, use the download URL
        self.url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        # Alternative URL that works for large files
        self.api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        self.download_url = f"https://drive.usercontent.google.com/uc?id={file_id}&export=download"
        
    def _get_file_info(self) -> dict:
        """Get file metadata from Google Drive API."""
        # Using the public API endpoint without authentication for public files
        response = requests.get(f"{self.api_url}?key=AIzaSyD8xWj2hP8sK9qy2LdJ7zXgY3r1q5s7t9")
        if response.status_code == 404:
            # Try the fallback method
            response = requests.head(self.url)
            response.raise_for_status()
            content_disposition = response.headers.get('content-disposition', '')
            file_name = content_disposition.split('filename=')[-1].strip('"') if 'filename=' in content_disposition else f"{self.file_id}.file"
            return {"name": file_name, "size": int(response.headers.get('content-length', 0))}
        
        if response.status_code != 200:
            # Fallback to a more reliable method
            response = requests.get(f"https://drive.google.com/uc?id={self.file_id}&export=download")
            response.raise_for_status()
            # Try to extract filename from content-disposition
            content_disposition = response.headers.get('content-disposition', '')
            file_name = content_disposition.split('filename=')[-1].strip('"') if 'filename=' in content_disposition else f"{self.file_id}.file"
            return {"name": file_name, "size": int(response.headers.get('content-length', 0))}
        
        data = response.json()
        return {"name": data.get("name", f"{self.file_id}.file"), "size": int(data.get("size", 0))}

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from Google Drive. Supports selecting specific instance_ids from archives."""
        output_path.mkdir(parents=True, exist_ok=True)
        file_info = self._get_file_info()
        file_name = file_info["name"]
        
        # Get the actual download URL that works
        actual_url = f"https://drive.usercontent.google.com/download?id={self.file_id}&export=download&confirm=t"
        
        # Handle the case where Google Drive returns a confirmation page
        session = requests.Session()
        response = session.get(actual_url, stream=True)
        
        if "confirm" in response.url and "logout" not in response.url:
            # Extract the confirmation token
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(response.url)
            confirm = parse_qs(parsed.query).get('confirm', [''])[0]
            if confirm:
                actual_url = f"https://drive.usercontent.google.com/download?id={self.file_id}&export=download&confirm={confirm}"
                response = session.get(actual_url, stream=True)

        if instance_ids is None:
            logger.info(
                f"Downloading file from Google Drive ID {self.file_id}"
            )
            _download_http_file(
                url=actual_url,
                out_file=output_path / file_name,
                desc=file_name,
            )
            return None

        remaining = set(instance_ids)
        logger.info(
            f"Downloading {len(remaining)} files from Google Drive ID {self.file_id}"
        )
        
        if file_name in remaining:
            _download_http_file(
                url=actual_url,
                out_file=output_path / file_name,
                desc=file_name,
            )
            remaining.remove(file_name)

        if RemoteArchive.is_supported_archive(file_name) and remaining:
            # Use the confirmed URL for archive extraction
            archive = RemoteArchive(actual_url, Path(file_name).suffix)
            extracted = archive.extract(
                filenames=sorted(remaining), output_path=output_path
            )
            remaining -= set(extracted)

        if remaining:
            msg = (
                f"Instance IDs {sorted(remaining)} not found in Google Drive source"
            )
            logger.warning(msg)

    @property
    def size(self) -> float:
        """Get file size in GB."""
        try:
            file_info = self._get_file_info()
            size = float(file_info.get("size", 0))
            return round(size / 1000 / 1000 / 1000, 2)  # convert to GB
        except Exception:
            # Fallback: try to get size from direct download
            try:
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    size = float(r.headers.get("content-length", 0))
                    return round(size / 1000 / 1000 / 1000, 2)
            except Exception:
                return 0.0

    @property
    def members(self) -> list[str]:
        """Get all members, including contents of archives."""
        file_info = self._get_file_info()
        file_name = file_info["name"]
        
        if RemoteArchive.is_supported_archive(file_name):
            try:
                # Get the actual download URL with confirmation if needed
                actual_url = f"https://drive.usercontent.google.com/download?id={self.file_id}&export=download&confirm=t"
                session = requests.Session()
                response = session.get(actual_url, stream=True)
                
                if "confirm" in response.url and "logout" not in response.url:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(response.url)
                    confirm = parse_qs(parsed.query).get('confirm', [''])[0]
                    if confirm:
                        actual_url = f"https://drive.usercontent.google.com/download?id={self.file_id}&export=download&confirm={confirm}"
                
                return RemoteArchive(actual_url, Path(file_name).suffix).members
            except Exception:
                return [file_name]
        
        return [file_name]

