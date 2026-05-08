from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests
import s3fs
from huggingface_hub import hf_hub_url, snapshot_download
from tqdm import tqdm
from tqdm.auto import tqdm as _tqdm

from imgnet.download.base import BaseDownloader
from imgnet.download.utils import _download_http_file
from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.utils import RemoteArchive, get_idc_client


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
        if hasattr(info, "usedStorage"):
            return round(
                float(info.usedStorage) / 1000 / 1000 / 1000, 2
            )  # convert to GB
        else:
            return 0.0

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
