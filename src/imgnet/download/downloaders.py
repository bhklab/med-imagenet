import os
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

import requests
import s3fs
from tqdm import tqdm
from tqdm.auto import tqdm as _tqdm

from imgnet.download.base import BaseDownloader
from imgnet.download.utils import (
    _TIMEOUT,
    _contains_subpath,
    _download_http_file,
    _http_session,
    archive_instance_ids,
    matching_instance_ids,
)
from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.utils import RemoteArchive, get_idc_client


os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_url, snapshot_download, HfApi


def _url_relative_path(url: str) -> PurePosixPath:
    """Return the URL path without the leading slash."""
    path = PurePosixPath(urlparse(url).path)
    if path.is_absolute():
        return PurePosixPath(*path.parts[1:])
    return path


def _extract_archive_ids(
    url: str,
    archive_path: str,
    instance_ids: set[str],
    output_path: Path,
) -> set[str]:
    """Extract instance ids that belong to ``archive_path``."""
    matched = matching_instance_ids(archive_path, instance_ids)
    if not matched:
        return set()
    archive = RemoteArchive(
        url,
        RemoteArchive.require_extension(archive_path),
        name=PurePosixPath(archive_path).name,
    )
    extracted = archive.extract(
        filenames=sorted(matched), output_path=output_path
    )
    return set(extracted)


def _member_names(file_name: str, inner_members: list[str]) -> list[str]:
    """Return archive members without a repeated archive stem."""
    stem = RemoteArchive.archive_stem(file_name)
    names: list[str] = []
    for member in inner_members:
        member_path = PurePosixPath(member)
        if member_path.parts and member_path.parts[0] == stem:
            remainder = PurePosixPath(*member_path.parts[1:])
            names.append(remainder.as_posix() if remainder.parts else member)
        else:
            names.append(member)
    return names


def _download_single_url(
    url: str,
    output_path: Path,
    instance_ids: list[str] | None,
    source_label: str,
) -> None:
    """Download one URL into ``output_path`` using the file's own name.

    The URL is the whole source, so directory segments in the URL are not
    recreated. A requested archive member is written at
    ``output_path/<instance id>``. The id may be the bare member path
    (``s0001/mri.nii.gz``) or start with the archive stem
    (``CancerVerse_Label/CancerVerse/...``); the index uses both.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    file_name = _url_relative_path(url).name
    destination = output_path / file_name

    if instance_ids is None:
        logger.info(f"Downloading all files from {source_label} {url}")
        _download_http_file(url=url, out_file=destination, desc=file_name)
        return

    remaining = set(instance_ids)
    logger.info(
        f"Downloading {len(remaining)} files from {source_label} {url}"
    )
    if file_name in remaining:
        _download_http_file(url=url, out_file=destination, desc=file_name)
        remaining.remove(file_name)

    if RemoteArchive.is_supported_archive(file_name) and remaining:
        archive = RemoteArchive(
            url, RemoteArchive.require_extension(file_name)
        )
        extracted = archive.extract(
            filenames=sorted(remaining),
            output_path=output_path,
            keep_stem=False,
        )
        remaining -= set(extracted)

    if remaining:
        logger.warning(
            f"Instance IDs {sorted(remaining)} not found in {source_label}"
        )


def _single_url_members(url: str) -> list[str]:
    file_name = _url_relative_path(url).name
    if not file_name:
        return []
    if RemoteArchive.is_supported_archive(file_name):
        archive = RemoteArchive(
            url, RemoteArchive.require_extension(file_name)
        )
        members = archive.members
        return list(
            dict.fromkeys(
                [
                    file_name,
                    *_member_names(file_name, members),
                    *archive_instance_ids(file_name, members),
                ]
            )
        )
    return [file_name]


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

        archives_needed: dict[str, set[str]] = {}

        if instance_ids is None:
            logger.info(
                f"Downloading all instances from Hugging Face repository {self.repo_id}"
            )
            files_to_download = self._top_level_members
            remaining: set[str] = set(files_to_download)
        else:
            logger.info(
                f"Downloading {len(instance_ids)} instances from Hugging Face repository {self.repo_id}"
            )
            files_to_download = []
            remaining = set(instance_ids)

            for file_name in self._top_level_members:
                if file_name in remaining:
                    files_to_download.append(file_name)
                    remaining.discard(file_name)
                    continue

                if RemoteArchive.is_supported_archive(file_name) and remaining:
                    archive_path = PurePosixPath(
                        RemoteArchive.strip_extension(file_name)
                    )

                    matched: set[str] = set()
                    for instance_id in remaining:
                        if _contains_subpath(instance_id, archive_path):
                            archives_needed.setdefault(file_name, set()).add(
                                instance_id
                            )
                            matched.add(instance_id)

                    remaining -= matched

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in Hugging Face repository {self.repo_id}"
                logger.warning(msg)

        for file_name, instance_paths in archives_needed.items():
            archive_url = hf_hub_url(
                repo_id=self.repo_id,
                filename=file_name,
                repo_type="dataset",
            )
            archive = RemoteArchive(
                archive_url,
                RemoteArchive.require_extension(file_name),
            )
            extracted = archive.extract(
                filenames=sorted(instance_paths), output_path=output_path
            )

            missing = instance_paths - set(extracted)
            if missing:
                logger.warning(
                    f"Instance IDs {sorted(missing)} not found in {file_name}"
                )

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
        return round(
            float(used_storage) / 1000 / 1000 / 1000, 2
        )  # convert to GB

    @property
    def members(self) -> list[str]:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(self.repo_id)
        siblings = info.siblings or []
        top_level = [sibling.rfilename for sibling in siblings]

        members: list[str] = []
        for file_name in top_level:
            if not RemoteArchive.is_supported_archive(file_name):
                members.append(file_name)
                continue

            # Keep the archive itself as a member (so you can still request it directly).
            members.append(file_name)

            archive_url = hf_hub_url(
                repo_id=self.repo_id,
                filename=file_name,
                repo_type="dataset",
            )
            archive = RemoteArchive(
                archive_url,
                RemoteArchive.require_extension(file_name),
            )
            try:
                members.extend(
                    archive_instance_ids(file_name, archive.members)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to list members of {file_name}: {e}")

        return members

    @property
    def _top_level_members(self) -> list[str]:
        """List top-level filenames in the repo without opening any archives."""

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
                    remaining -= _extract_archive_ids(
                        file_info["links"]["self"],
                        file_name,
                        remaining,
                        output_path,
                    )

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
        session = _http_session()
        resp = session.get(f"{self.url}/{self.record_id}", timeout=_TIMEOUT)
        resp.raise_for_status()
        files = resp.json()["files"]
        if len(files) == 0:
            msg = f"No files found for Zenodo record {self.record_id}"
            raise FileNotFoundError(msg)

        return files

    @property
    def size(self) -> float:
        size = float(sum(f["size"] for f in self.files_info))
        return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        updated_file_names = []

        for file_info in self.files_info:
            file_name = file_info["key"]
            updated_file_names.append(file_name)
            if RemoteArchive.is_supported_archive(file_name):
                archive = RemoteArchive(
                    file_info["links"]["self"],
                    RemoteArchive.require_extension(file_name),
                    name=PurePosixPath(file_name).name,
                )
                updated_file_names.extend(
                    archive_instance_ids(file_name, archive.members)
                )

        return list(dict.fromkeys(updated_file_names))


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
        _download_single_url(
            self.url, output_path, instance_ids, "Dropbox source"
        )

    @property
    def size(self) -> float:
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            size = float(r.headers.get("content-length", 0))
            return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        return _single_url_members(self.url)


class HttpDownloader(BaseDownloader):
    def __init__(self, url: str) -> None:
        self.url = url

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Download from a direct HTTP(S) URL. Supports selecting instance_ids from archives."""
        _download_single_url(
            self.url, output_path, instance_ids, "HTTP source"
        )

    @property
    def size(self) -> float:
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            size = float(r.headers.get("content-length", 0))
            return round(size / 1000 / 1000 / 1000, 2)  # convert to GB

    @property
    def members(self) -> list[str]:
        return _single_url_members(self.url)


class S3Downloader(BaseDownloader):
    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self.fs = s3fs.S3FileSystem(anon=True)

    def _object_key(self, file_path: str) -> str:
        path = file_path.removeprefix("s3://")
        prefix = f"{self.bucket_name}/"
        if path.startswith(prefix):
            return path[len(prefix) :]
        return path

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
                key = self._object_key(file_path)
                file_name = Path(key).name
                if (
                    key in remaining
                    or file_path in remaining
                    or file_name in remaining
                ):
                    files_to_download.append(file_path)
                    remaining.discard(key)
                    remaining.discard(file_path)
                    remaining.discard(file_name)
                    continue

                if RemoteArchive.is_supported_archive(key) and remaining:
                    remaining -= _extract_archive_ids(
                        f"s3://{file_path}",
                        key,
                        remaining,
                        output_path,
                    )

            if remaining:
                msg = f"Instance IDs {sorted(remaining)} not found in S3 bucket {self.bucket_name}"
                logger.warning(msg)

        for file_path in files_to_download:
            size = self.fs.info(file_path)["size"]
            out_file = output_path / self._object_key(file_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)

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
        expanded_members: list[str] = []
        for file_path in self.fs.find(self.bucket_name):
            key = self._object_key(file_path)
            expanded_members.append(key)
            if RemoteArchive.is_supported_archive(key):
                archive = RemoteArchive(
                    f"s3://{file_path}",
                    RemoteArchive.require_extension(key),
                )
                expanded_members.extend(
                    archive_instance_ids(key, archive.members)
                )
        return list(dict.fromkeys(expanded_members))


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
            logger.info(
                f"Downloading all files from GitHub repository {self.repo_id}"
            )
            files_to_download = self._repository_files()
        else:
            logger.info(
                f"Downloading {len(instance_ids)} instances from GitHub repository {self.repo_id}"
            )
            remaining = set(instance_ids)
            files_to_download = []

            for file_path in self._repository_files():
                if file_path in remaining:
                    files_to_download.append(file_path)
                    remaining.remove(file_path)
                    continue

                if RemoteArchive.is_supported_archive(file_path) and remaining:
                    remaining -= _extract_archive_ids(
                        self._get_raw_url(file_path),
                        file_path,
                        remaining,
                        output_path,
                    )

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
                total_size = int(response.headers.get("content-length", 0))

                # Download with progress
                with open(local_file_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {Path(file_path).name}",
                        leave=False,
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

    def _repository_files(self) -> list[str]:
        """Return blob paths in the repository, without opening archives."""
        cache_key = "files"
        if cache_key in self._cache:
            return self._cache[cache_key]

        branch = self._get_default_branch()
        url = f"{self._api_base}/{self.repo_id}/git/trees/{branch}?recursive=1"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            files = [
                item["path"]
                for item in data.get("tree", [])
                if item.get("type") == "blob"
            ]
            self._cache[cache_key] = files
            return files
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get repository members: {e}")
            return []

    @property
    def members(self) -> list[str]:
        """Return selectable instance ids, including files inside archives."""
        cache_key = "members"
        if cache_key in self._cache:
            return self._cache[cache_key]

        expanded: list[str] = []
        for file_path in self._repository_files():
            expanded.append(file_path)
            if not RemoteArchive.is_supported_archive(file_path):
                continue
            try:
                archive = RemoteArchive(
                    self._get_raw_url(file_path),
                    RemoteArchive.require_extension(file_path),
                )
                expanded.extend(
                    archive_instance_ids(file_path, archive.members)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to list members of {file_path}: {e}")

        self._cache[cache_key] = expanded
        return expanded

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
            self._size = round(
                sum(downloader.size for downloader in self.downloaders), 2
            )
        return self._size

    def download(
        self,
        output_path: Path,
        instance_ids: list[str] | None = None,
        **kwargs: Any,
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
                logger.info(
                    f"members for {type(self.downloaders)}: \n {members}"
                )
                logger.info(f"ids accepted: {ids}")
                if len(ids) > 0:
                    downloader.download(
                        output_path, instance_ids=ids, **kwargs
                    )
