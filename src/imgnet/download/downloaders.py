import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import s3fs
from tqdm import tqdm
from tqdm.auto import tqdm as _tqdm

import tempfile
import shutil

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
        self.url = "https://zenodo.org/api/records"

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
        resp = requests.get(f"{self.url}/{self.record_id}")
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


class LMUMunichDownloader(BaseDownloader):
    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        self.url = "https://fdat.uni-tuebingen.de/api/records"

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download files from Ludwig-Maximilians-University Munich. Here instance_ids is a list of filenames to download."""
        output_path.mkdir(parents=True, exist_ok=True)
        files = self.files_info

        if instance_ids is None:
            files_to_download = files
            logger.info(
                f"Downloading all files from LMU Munich record {self.record_id}"
            )
        else:
            logger.info(
                f"Downloading {len(instance_ids)} files from LMU Munich record {self.record_id}"
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
                msg = f"Instance IDs {sorted(remaining)} not found in LMU Munich record {self.record_id}"
                logger.warning(msg)
        for file_info in files_to_download:
            _download_http_file(
                url=file_info["links"]["content"],
                out_file=output_path / file_info["key"],
                desc=file_info["key"],
                size=file_info["size"],
            )

    @property
    def files_info(self) -> list[dict]:
        resp = requests.get(f"{self.url}/{self.record_id}/files")
        resp.raise_for_status()
        if len(resp.json()["entries"]) == 0:
            msg = f"No files found for Zenodo record {self.record_id}"
            raise FileNotFoundError(msg)

        return resp.json()["entries"]

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
        """
        Initialize GitHub downloader.
        
        Args:
            repo_id: GitHub repository in format "owner/repo"
        """
        self.repo_id = repo_id
        self._api_base = "https://api.github.com/repos"
        self._raw_base = "https://raw.githubusercontent.com"
        self._cache = {}  # Simple cache for API responses
        
    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Download from GitHub repository."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        if instance_ids is None:
            logger.info(f"Downloading all files from GitHub repository {self.repo_id}")
            files_to_download = self.members
        else:
            logger.info(
                f"Downloading {len(instance_ids)} instances from GitHub repository {self.repo_id}"
            )
            remaining = set(instance_ids)
            files_to_download = []
            
            # Get all members first to check for archives
            all_members = self.members
            
            for file_path in all_members:
                if file_path in remaining:
                    files_to_download.append(file_path)
                    remaining.remove(file_path)
                    continue
                
                # Check if this is an archive that might contain remaining files
                if RemoteArchive.is_supported_archive(file_path) and remaining:
                    # Try to extract and see if it contains any of the remaining files
                    archive_url = self._get_raw_url(file_path)
                    archive = RemoteArchive(archive_url, Path(file_path).suffix)
                    
                    # Create temp dir for extraction
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        extracted = archive.extract(
                            filenames=sorted(remaining),
                            output_path=temp_path,
                        )
                        
                        # If we extracted something, move it to the actual output
                        if extracted:
                            for extracted_file in extracted:
                                src = temp_path / extracted_file
                                dst = output_path / extracted_file
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(src), str(dst))
                            remaining -= set(extracted)
            
            if remaining:
                logger.warning(
                    f"Instance IDs {sorted(remaining)} not found in GitHub repository {self.repo_id}"
                )
        
        # Download all collected files
        if files_to_download:
            self._download_files(files_to_download, output_path)
    
    def _download_files(self, files: list[str], output_path: Path) -> None:
        """Download a list of files from GitHub preserving directory structure."""
        
        for file_path in tqdm(files, desc="Downloading files"):
            local_file_path = output_path / file_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get the raw content URL
            raw_url = self._get_raw_url(file_path)
            
            try:
                # Stream download for large files
                response = requests.get(raw_url, stream=True)
                response.raise_for_status()
                
                # Get total size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress
                with open(local_file_path, 'wb') as f:
                    with tqdm(
                        total=total_size, 
                        unit='B', 
                        unit_scale=True,
                        desc=f"Downloading {Path(file_path).name}",
                        leave=False
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {file_path}: {e}")
                # Remove partially downloaded file
                if local_file_path.exists():
                    local_file_path.unlink()
                raise
    
    def _get_raw_url(self, file_path: str) -> str:
        """Get the raw content URL for a file."""
        # First, get the default branch
        branch = self._get_default_branch()
        return f"{self._raw_base}/{self.repo_id}/{branch}/{file_path}"
    
    def _get_default_branch(self) -> str:
        """Get the default branch of the repository."""
        cache_key = "default_branch"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        url = f"{self._api_base}/{self.repo_id}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        branch = data.get("default_branch", "main")
        self._cache[cache_key] = branch
        return branch
    
    @property
    def size(self) -> float:
        """Return the size of the repository in GB."""
        cache_key = "size"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        url = f"{self._api_base}/{self.repo_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # GitHub API returns size in KB
            size_kb = data.get("size", 0)
            size_gb = size_kb / 1000 / 1000  # Convert KB to GB
            
            result = round(size_gb, 2)
            self._cache[cache_key] = result
            return result
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get repository size: {e}")
            return 0.0
    
    @property
    def members(self) -> list[str]:
        """Return all file paths in the repository."""
        cache_key = "members"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get the default branch
        branch = self._get_default_branch()
        
        # Get the git tree recursively
        url = f"{self._api_base}/{self.repo_id}/git/trees/{branch}?recursive=1"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Filter out directories, keep only files
            members = [
                item["path"] 
                for item in data.get("tree", [])
                if item.get("type") == "blob"
            ]
            
            self._cache[cache_key] = members
            return members
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get repository members: {e}")
            return []
    
    def _get_file_size(self, file_path: str) -> int:
        """Get the size of a specific file."""
        # First try to get from members list with sizes
        url = f"{self._api_base}/{self.repo_id}/contents/{file_path}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("size", 0)
        except requests.exceptions.RequestException:
            return 0

class CompositeDownloader(BaseDownloader):
    def __init__(self, downloaders: list[BaseDownloader]):
        self.downloaders = downloaders
        self._size = None
        self._members = None
    @property
    def members(self) -> list[str]:
        if self._members is None:
            result = []
            for downloader in self.downloaders:
                result.extend(downloader.members)
            self._members = result
        return self._members  # Always return, even if empty
    @property
    def size(self) -> float:
        if self._size is None:
            self._size = sum(downloader.size for downloader in self.downloaders)
        return self._size
    def download(self,
            output_path: Path,
            instance_ids: list[str] | None = None,
            **kwargs: Any
        ) -> None:
        """
        Download from all sources.
        
        If instance_ids is provided, each downloader will only download files
        that match the requested IDs. IDs not found in any source are ignored.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        if instance_ids is None:
            for downloader in self.downloaders:
                downloader.download(output_path, **kwargs)
        else:
            for downloader in self.downloaders:
                members = downloader.members
                ids = [id for id in instance_ids if id in members]
                if len(ids) > 0:
                    downloader.download(output_path, instance_ids=ids, **kwargs)

        


