import tarfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, cast
from urllib.parse import urlparse

import fsspec
from tqdm import tqdm

from imgnet.loggers import logger

SUPPORTED_EXTENSIONS = [".zip", ".tar", ".tar.gz", ".tgz"]
TAR_EXTENSIONS = {".tar", ".tar.gz", ".tgz"}


class RemoteArchive:
    def __init__(self, url: str, archive_extension: str) -> None:
        self.url = url

        if archive_extension not in SUPPORTED_EXTENSIONS:
            msg = f"Invalid archive extension: {archive_extension}"
            raise ValueError(msg)
        self.archive_extension = archive_extension

    @classmethod
    def archive_extension_from(cls, path_or_url: str) -> str | None:
        """Return the supported archive extension for a path/URL, or None."""
        name = Path(urlparse(path_or_url).path).name.lower()
        for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
            if name.endswith(ext):
                return ext
        return None

    @classmethod
    def require_extension(cls, path_or_url: str) -> str:
        """Return the supported archive extension, or raise if unsupported."""
        ext = cls.archive_extension_from(path_or_url)
        if ext is None:
            msg = f"Unsupported archive: {path_or_url}"
            raise ValueError(msg)
        return ext

    @classmethod
    def archive_stem(cls, path_or_url: str) -> str:
        """Return the archive filename stem, handling compound suffixes like .tar.gz."""
        name = Path(urlparse(path_or_url).path).name
        name_lower = name.lower()
        for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
            if name_lower.endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem

    @property
    def members(self) -> list[str]:
        if self.archive_extension == ".zip":
            return self._members_zip()
        if self.archive_extension in TAR_EXTENSIONS:
            return self._members_tar()
        return []

    @classmethod
    def is_supported_archive(cls, url_or_filename: str) -> bool:
        return cls.archive_extension_from(url_or_filename) is not None

    @property
    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    @property
    def archive_file(self) -> Path:
        return Path(urlparse(self.url).path)

    @contextmanager
    def _open_archive(self) -> Iterator[IO[bytes]]:
        storage_options = (
            {"anon": True} if self.url.startswith("s3://") else {}
        )
        with fsspec.open(
            self.url, mode="rb", **storage_options
        ).open() as remote_file:
            yield cast("IO[bytes]", remote_file)

    def _extract_zip(
        self,
        filenames: list[str] | None = None,
        output_path: Path | None = None,
    ) -> list[str]:
        extracted: list[str] = []
        with self._open_archive() as archive_obj:  # noqa: SIM117
            with zipfile.ZipFile(archive_obj, "r") as zip_ref:
                if filenames:
                    logger.info(f"Extracting {filenames} from {self.url}")
                    for filename in filenames:
                        try:
                            zip_ref.extract(filename, output_path)
                            extracted.append(filename)
                        except KeyError:
                            continue
                else:
                    logger.info(f"Extracting all files from {self.url}")
                    zip_ref.extractall(output_path)
                    extracted = self._members_zip()
        return extracted

    def _tar_check_root(self, filename: str) -> bool:
        """To speed up the extraction, we can check if the filename root is the same as the url filename stem.
        If it is, we can skip the extraction.
        """
        url_stem = self.archive_stem(str(self.archive_file))
        return Path(filename).parts[0] == url_stem

    def check_tar_filenames(self, filenames: list[str]) -> list[str]:
        """Check if the filenames roots are the same as the url filename stem.
        If they are, we can skip the extraction.
        """
        return [
            filename
            for filename in filenames
            if self._tar_check_root(filename)
        ]

    def _extract_tar(
        self,
        filenames: list[str] | None = None,
        output_path: Path | None = None,
    ) -> list[str]:
        # WARNING: This is a BIG assumption to speed up the extraction.
        # Only extract the files that the filename roots are the same as the url filename stem.
        # For example, if the file name is "Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
        # Only extract a RemoteArcive that looks like "s3://msd-for-monai/Task01_BrainTumour.tar".
        if filenames:
            filenames_to_extract = self.check_tar_filenames(filenames)
            if len(filenames_to_extract) == 0:
                return []
            filenames = filenames_to_extract

        dest = output_path if output_path is not None else Path()
        extracted: list[str] = []
        with self._open_archive() as archive_obj:  # noqa: SIM117
            with tarfile.open(fileobj=archive_obj, mode="r|*") as tar_ref:
                if filenames:
                    target_names = set(filenames)
                    logger.info(f"Extracting {filenames} from {self.url}")
                    for member in tar_ref:
                        if member.name in target_names:
                            tar_ref.extract(member, dest)
                            extracted.append(member.name)
                            if len(extracted) == len(target_names):
                                break
                else:
                    logger.info(f"Extracting all files from {self.url}")
                    for member in tar_ref:
                        tar_ref.extract(member, dest)
                        if member.isfile():
                            extracted.append(member.name)
        return extracted

    def extract(
        self,
        filenames: list[str] | None = None,
        output_path: Path | None = None,
    ) -> list[str]:
        if self.archive_extension == ".zip":
            return self._extract_zip(filenames, output_path)
        if self.archive_extension in TAR_EXTENSIONS:
            return self._extract_tar(filenames, output_path)
        return []

    def _members_zip(self) -> list[str]:
        with self._open_archive() as archive_obj:  # noqa: SIM117
            with zipfile.ZipFile(archive_obj, "r") as zip_ref:
                return [
                    info.filename
                    for info in zip_ref.infolist()
                    if not info.is_dir()
                ]

    def _members_tar(self) -> list[str]:
        logger.warning(
            "Listing members from tar file, this may take a while..."
        )
        with self._open_archive() as archive_obj:  # noqa: SIM117
            with tarfile.open(fileobj=archive_obj, mode="r|*") as tar_ref:
                return [
                    member.name
                    for member in tqdm(
                        tar_ref,
                        desc=f"Listing {self.archive_file.name}",
                        unit="member",
                    )
                    if member.isfile()
                ]
