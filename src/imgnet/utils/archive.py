import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path, PurePosixPath
from typing import IO, cast
from urllib.parse import urlparse

import fsspec
import aiohttp
from tqdm import tqdm

from imgnet.loggers import logger

SUPPORTED_EXTENSIONS = [".zip", ".tar", ".tar.gz", ".tgz"]
TAR_EXTENSIONS = {".tar", ".tar.gz", ".tgz"}


class RemoteArchive:
    def __init__(
        self, url: str, archive_extension: str, name: str | None = None
    ) -> None:
        """Wrap a remote archive.

        ``name`` is the archive filename. Pass it when the URL does not end
        in the filename, such as Zenodo ``.../archive.zip/content`` links.
        """
        self.url = url
        self.name = name or Path(urlparse(url).path).name

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
        ext = cls.archive_extension_from(name)
        if ext is None:
            return Path(name).stem
        return name[: -len(ext)]

    @classmethod
    def strip_extension(cls, filename: str) -> str:
        filename = str(filename)
        for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
            if filename.endswith(ext):
                return filename[: -len(ext)]
        return filename

    @property
    def stem(self) -> str:
        return self.archive_stem(self.name)

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
    def _open_archive(
        self, *, streaming: bool = False
    ) -> Iterator[IO[bytes]]:
        """Open the archive for reading.
        ``streaming`` reads HTTP(S) bodies with one GET instead of a range
        request per block. Only sequential readers (tar) can use it; zip
        needs to seek to its central directory.
        """
        storage_options: dict[str, object] = {}
        if self.url.startswith("s3://"):
            storage_options["anon"] = True
        elif streaming and self.url.startswith(("http://", "https://")):
            storage_options["block_size"] = 0
            storage_options["client_kwargs"] = {
                "timeout": aiohttp.ClientTimeout(
                    total=None, sock_connect=30, sock_read=300
                )
            }
        with fsspec.open(
            self.url, mode="rb", **storage_options
        ).open() as remote_file:
            yield cast("IO[bytes]", remote_file)

    @contextmanager
    def _open_zip(self) -> Iterator[zipfile.ZipFile]:
        """Open the zip in place, or spool it when the server cannot seek.

        Servers with range requests (Zenodo, Hugging Face, S3) only send the
        central directory and the members that are read. Servers without
        them make ``ZipFile`` raise, so the whole body is copied first.
        """
        with self._open_archive() as remote:
            try:
                zip_ref = zipfile.ZipFile(remote)
            except (ValueError, OSError, zipfile.BadZipFile) as e:
                if not self.url.startswith(("http://", "https://")):
                    raise
                logger.info(
                    f"{self.url} does not support seeking ({e}); "
                    "downloading the whole zip first"
                )
                zip_ref = None
            if zip_ref is not None:
                with zip_ref:
                    yield zip_ref
                return
        with (
            self._open_archive() as remote,
            tempfile.TemporaryFile() as spool,
        ):
            shutil.copyfileobj(remote, spool)
            spool.seek(0)
            with zipfile.ZipFile(spool) as zip_ref:
                yield zip_ref

    def _target_name(self, filename: str, stem: str) -> str:
        """Return the member path, relative to the stem, for an instance id."""
        path = PurePosixPath(filename)
        stem_path = PurePosixPath(stem)
        if stem and path.is_relative_to(stem_path):
            remainder = path.relative_to(stem_path)
            return remainder.as_posix() if remainder.parts else ""
        if stem and stem in path.parts:
            tail = path.parts[path.parts.index(stem) + 1 :]
            return PurePosixPath(*tail).as_posix() if tail else ""
        return filename

    def _relative_member(self, member_name: str) -> str:
        """Return a member path without a leading folder named after the archive.

        Some archives store ``imagesTr/a.nii.gz`` and others store
        ``Task04/imagesTr/a.nii.gz`` inside ``Task04.tar``. Both map to
        ``imagesTr/a.nii.gz``.
        """
        path = PurePosixPath(member_name)
        if len(path.parts) > 1 and path.parts[0] == self.stem:
            return PurePosixPath(*path.parts[1:]).as_posix()
        return path.as_posix()

    def _targets(self, filenames: list[str] | None) -> dict[str, str] | None:
        """Map member paths relative to the stem back to requested ids."""
        if not filenames:
            return None
        targets: dict[str, str] = {}
        for filename in filenames:
            target_name = self._target_name(filename, self.stem)
            if target_name:
                targets[target_name] = filename
        return targets

    def _extract_destination(
        self, output_path: Path | None, *, keep_stem: bool
    ) -> Path:
        base = output_path if output_path is not None else Path()
        return base / self.stem if keep_stem else base

    def _extracted_id(self, relative: str, *, keep_stem: bool) -> str:
        if keep_stem:
            return (PurePosixPath(self.stem) / relative).as_posix()
        return relative

    @staticmethod
    def _write_member(source: IO[bytes], dest: Path, relative: str) -> bool:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            logger.warning(f"Skipping unsafe archive member {relative}")
            return False
        out_file = dest.joinpath(*path.parts)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("wb") as out:
            shutil.copyfileobj(source, out)
        return True

    def _extract_entries(
        self,
        entries: Iterator[tuple[str, Callable[[], IO[bytes]]]],
        filenames: list[str] | None,
        output_path: Path | None,
        *,
        keep_stem: bool,
    ) -> list[str]:
        """Write archive entries and return the ids that were written.

        A requested id is written at ``output_path / id``, so the saved path
        is the query path whether or not the id starts with the archive
        stem. Without ids, every member goes under ``output_path / stem``
        (``keep_stem``) or straight under ``output_path``.
        """
        base = output_path if output_path is not None else Path()
        pending = self._targets(filenames)
        extracted: list[str] = []
        for name, open_member in entries:
            relative = self._relative_member(name)
            if pending is None:
                dest = self._extract_destination(
                    output_path, keep_stem=keep_stem
                )
                with open_member() as source:
                    if self._write_member(source, dest, relative):
                        extracted.append(
                            self._extracted_id(relative, keep_stem=keep_stem)
                        )
                continue
            if relative not in pending:
                continue
            instance_id = pending[relative]
            with open_member() as source:
                if not self._write_member(source, base, instance_id):
                    continue
            del pending[relative]
            extracted.append(instance_id)
            if not pending:
                break
        return extracted

    def _log_extract(self, filenames: list[str] | None) -> None:
        if filenames:
            logger.info(f"Extracting {filenames} from {self.url}")
        else:
            logger.info(f"Extracting all files from {self.url}")

    def _extract_zip(
        self,
        filenames: list[str] | None = None,
        output_path: Path | None = None,
        *,
        keep_stem: bool = True,
    ) -> list[str]:
        self._log_extract(filenames)
        with self._open_zip() as zip_ref:
            entries = (
                (info.filename, partial(zip_ref.open, info))
                for info in zip_ref.infolist()
                if not info.is_dir()
            )
            return self._extract_entries(
                entries, filenames, output_path, keep_stem=keep_stem
            )

    def _tar_check_root(self, filename: str) -> bool:
        """To speed up the extraction, we can check if the filename root is the same as the url filename stem.
        If it is, we can skip the extraction.
        """
        return Path(filename).parts[0] == self.stem

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
        *,
        keep_stem: bool = True,
    ) -> list[str]:
        self._log_extract(filenames)

        with self._open_archive(streaming=True) as archive_obj:  # noqa: SIM117

            with tarfile.open(fileobj=archive_obj, mode="r|*") as tar_ref:
                entries = (
                    (
                        member.name,
                        cast(
                            "Callable[[], IO[bytes]]",
                            partial(tar_ref.extractfile, member),
                        ),
                    )
                    for member in tar_ref
                    if member.isfile()
                )
                return self._extract_entries(
                    entries, filenames, output_path, keep_stem=keep_stem
                )

    def extract(
        self,
        filenames: list[str] | None = None,
        output_path: Path | None = None,
        *,
        keep_stem: bool = True,
    ) -> list[str]:
        if self.archive_extension == ".zip":
            return self._extract_zip(
                filenames, output_path, keep_stem=keep_stem
            )
        if self.archive_extension in TAR_EXTENSIONS:
            return self._extract_tar(
                filenames, output_path, keep_stem=keep_stem
            )
        return []

    def _members_zip(self) -> list[str]:
        with self._open_zip() as zip_ref:
            return [
                info.filename
                for info in zip_ref.infolist()
                if not info.is_dir()
            ]

    def _members_tar(self) -> list[str]:
        logger.warning(
            f"Listing members from tar file {self.url}, this may take a while..."
        )
        with self._open_archive(streaming=True) as archive_obj:  # noqa: SIM117
            with tarfile.open(fileobj=archive_obj, mode="r|*") as tar_ref:
                out = [
                    f"{self.stem}/{self._relative_member(member.name)}"
                    for member in tqdm(
                        tar_ref,
                        desc=f"Listing {self.name}",
                        unit="member",
                    )
                    if member.isfile()
                ]

                return out
