"""Downloaders should write one shared fixture layout.

The fake tree contains three objects:

- ``root/zip/archive.zip`` with member ``zip_test_file.nii.gz``
- ``root/tar/archive.tar.gz`` with member ``tar_test_file.nii.gz``
- ``root/uncompressed/uncompressed_test_file.nii.gz``

Instance ids drop the archive extension. Multi-file sources receive all three
ids in one ``download`` call. HTTP and Dropbox are one URL each, so an archive
URL writes its member directly under the output directory.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import fsspec
import pytest
import requests
from fsspec.registry import get_filesystem_class, register_implementation
from fsspec.spec import AbstractFileSystem

from imgnet.download.base import BaseDownloader
from imgnet.download.downloaders import (
    CompositeDownloader,
    DropboxDownloader,
    GitHubDownloader,
    HttpDownloader,
    HuggingFaceDownloader,
    S3Downloader,
    ZenodoDownloader,
)
from imgnet.utils.archive import RemoteArchive

ZIP_BYTES = b"zip-test-file"
TAR_BYTES = b"tar-test-file"
LOOSE_BYTES = b"uncompressed-test-file"

REMOTE_FILES = (
    "root/zip/archive.zip",
    "root/tar/archive.tar.gz",
    "root/uncompressed/uncompressed_test_file.nii.gz",
)
INSTANCE_IDS = (
    "root/zip/archive/zip_test_file.nii.gz",
    "root/tar/archive/tar_test_file.nii.gz",
    "root/uncompressed/uncompressed_test_file.nii.gz",
)
_PAYLOADS = {
    "root/zip/archive/zip_test_file.nii.gz": ZIP_BYTES,
    "root/tar/archive/tar_test_file.nii.gz": TAR_BYTES,
    "root/uncompressed/uncompressed_test_file.nii.gz": LOOSE_BYTES,
}

_HF_REPO = "fake-org/fake-dataset"
_GITHUB_REPO = "fake-org/fake-dataset"
_ZENODO_RECORD = "1234567"
_S3_BUCKET = "fake-bucket"


def _build_source_tree(root: Path) -> None:
    loose = root / "root/uncompressed/uncompressed_test_file.nii.gz"
    loose.parent.mkdir(parents=True)
    loose.write_bytes(LOOSE_BYTES)

    zip_path = root / "root/zip/archive.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("zip_test_file.nii.gz", ZIP_BYTES)

    tar_path = root / "root/tar/archive.tar.gz"
    tar_path.parent.mkdir(parents=True)
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo(name="tar_test_file.nii.gz")
        info.size = len(TAR_BYTES)
        archive.addfile(info, io.BytesIO(TAR_BYTES))


def _url_for(root: Path, base_url: str, relative: str) -> str:
    """Archive URLs are local files so zip extraction can seek."""
    if RemoteArchive.is_supported_archive(relative):
        return (root / relative).resolve().as_uri()
    return f"{base_url}/{relative}"


def _assert_shared_layout(output: Path) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, payload in _PAYLOADS.items():
        downloaded = output / relative
        if not downloaded.is_file():
            missing.append(relative)
            continue
        if downloaded.read_bytes() != payload:
            mismatched.append(relative)
    assert not missing and not mismatched, (
        f"missing={missing}, mismatched={mismatched}"
    )


class _JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return None


def _handler(directory: str) -> type[SimpleHTTPRequestHandler]:
    class Handler(_QuietHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=directory, **kwargs)

    return Handler


class _FakeS3(AbstractFileSystem):
    protocol = "s3"
    root: Path
    bucket: str

    def __init__(self, anon: bool = True, **kwargs: object) -> None:
        super().__init__()

    def find(self, path: str, **kwargs: object) -> list[str]:
        found: list[str] = []
        for file in self.root.rglob("*"):
            if file.is_file():
                relative = file.relative_to(self.root).as_posix()
                found.append(f"{self.bucket}/{relative}")
        return found

    def info(self, path: str, **kwargs: object) -> dict[str, object]:
        local = self._local(path)
        return {"name": path, "size": local.stat().st_size, "type": "file"}

    def _open(self, path: str, mode: str = "rb", **kwargs: object):  # noqa: ANN202
        return self._local(path).open(mode)

    def _local(self, path: str) -> Path:
        key = self._strip_protocol(path)
        prefix = f"{self.bucket}/"
        if key.startswith(prefix):
            key = key[len(prefix) :]
        return self.root / key


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _build_source_tree(root)
    return root


@pytest.fixture
def source_url(source_tree: Path) -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler(str(source_tree))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def downloaders(
    source_tree: Path,
    source_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, BaseDownloader]]:
    def fake_hf_hub_url(
        repo_id: str, filename: str, repo_type: str = "dataset"
    ) -> str:
        return (source_tree / filename).resolve().as_uri()

    def fake_snapshot_download(
        repo_id: str,
        local_dir: str | Path,
        allow_patterns: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        for pattern in allow_patterns or []:
            destination = Path(local_dir) / pattern
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_tree / pattern, destination)

    class FakeHfApi:
        def dataset_info(self, repo_id: str, **kwargs: object) -> object:
            class Info:
                siblings = [
                    type("Sibling", (), {"rfilename": name})()
                    for name in REMOTE_FILES
                ]
                used_storage = 1

            return Info()

    def zenodo_link(relative: str) -> str:
        """Zenodo links end in ``/content``, not in the file name."""
        if not RemoteArchive.is_supported_archive(relative):
            return _url_for(source_tree, source_url, relative)
        content = (
            source_tree.parent / "zenodo" / "files" / relative / "content"
        )
        content.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_tree / relative, content)
        return content.resolve().as_uri()

    files = [
        {
            "key": relative,
            "size": (source_tree / relative).stat().st_size,
            "links": {"self": zenodo_link(relative)},
        }
        for relative in REMOTE_FILES
    ]

    class RoutingSession(requests.Session):
        def get(self, url: str, **kwargs: object) -> object:  # type: ignore[override]
            if str(url).startswith("https://zenodo.org/api/records/"):
                return _JsonResponse({"files": files})
            return super().get(url, **kwargs)

    real_get = requests.get

    def fake_get(url: str, *args: object, **kwargs: object) -> object:
        if url.startswith(f"https://api.github.com/repos/{_GITHUB_REPO}"):
            if "/git/trees/" in url:
                tree = [
                    {"path": relative, "type": "blob"}
                    for relative in REMOTE_FILES
                ]
                return _JsonResponse({"tree": tree})
            return _JsonResponse({"default_branch": "main", "size": 1})
        return real_get(url, *args, **kwargs)

    def fake_raw_url(self: GitHubDownloader, file_path: str) -> str:
        return _url_for(source_tree, source_url, file_path)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    monkeypatch.setattr("imgnet.download.downloaders.HfApi", FakeHfApi)
    monkeypatch.setattr(
        "imgnet.download.downloaders.hf_hub_url", fake_hf_hub_url
    )
    monkeypatch.setattr(
        "imgnet.download.downloaders.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        "imgnet.download.downloaders._http_session", RoutingSession
    )
    monkeypatch.setattr("imgnet.download.downloaders.requests.get", fake_get)
    monkeypatch.setattr(GitHubDownloader, "_get_raw_url", fake_raw_url)
    monkeypatch.setattr(
        "imgnet.download.downloaders.s3fs.S3FileSystem", _FakeS3
    )

    _FakeS3.root = source_tree
    _FakeS3.bucket = _S3_BUCKET
    original_s3 = get_filesystem_class("s3")
    register_implementation("s3", _FakeS3, clobber=True)

    zenodo = ZenodoDownloader(_ZENODO_RECORD)
    github = GitHubDownloader(_GITHUB_REPO)
    s3 = S3Downloader(_S3_BUCKET)
    huggingface = HuggingFaceDownloader(_HF_REPO)
    try:
        yield {
            "zenodo": zenodo,
            "github": github,
            "s3": s3,
            "huggingface": huggingface,
            "composite": CompositeDownloader(
                [zenodo, github, s3, huggingface]
            ),
        }
    finally:
        register_implementation("s3", original_s3, clobber=True)


@pytest.mark.parametrize(
    "source_name",
    ["zenodo", "github", "s3", "huggingface", "composite"],
)
def test_downloader_writes_shared_layout(
    source_name: str,
    downloaders: dict[str, BaseDownloader],
    tmp_path: Path,
) -> None:
    """Selected ids are written under the output directory."""
    output = tmp_path / "test_dir"
    downloaders[source_name].download(
        output, instance_ids=list(INSTANCE_IDS)
    )
    _assert_shared_layout(output)


@pytest.mark.parametrize("source_cls", [HttpDownloader, DropboxDownloader])
@pytest.mark.parametrize(
    ("relative", "member_name", "payload"),
    [
        ("root/zip/archive.zip", "zip_test_file.nii.gz", ZIP_BYTES),
        ("root/tar/archive.tar.gz", "tar_test_file.nii.gz", TAR_BYTES),
        # CancerVerse_Label.tar.gz is indexed as CancerVerse_Label/<member>.
        ("root/zip/archive.zip", "archive/zip_test_file.nii.gz", ZIP_BYTES),
        (
            "root/tar/archive.tar.gz",
            "archive/tar_test_file.nii.gz",
            TAR_BYTES,
        ),
        (
            "root/uncompressed/uncompressed_test_file.nii.gz",
            "uncompressed_test_file.nii.gz",
            LOOSE_BYTES,
        ),
    ],
)
def test_single_url_source_writes_its_file(
    source_cls: type[HttpDownloader],
    relative: str,
    member_name: str,
    payload: bytes,
    source_url: str,
    tmp_path: Path,
) -> None:
    """One HTTP or Dropbox URL writes each requested id at ``test_dir/<id>``."""
    output = tmp_path / "test_dir"
    downloader = source_cls(f"{source_url}/{relative}")
    assert member_name in downloader.members
    downloader.download(output, instance_ids=[member_name])
    downloaded = output / member_name
    assert downloaded.is_file(), member_name
    assert downloaded.read_bytes() == payload


@pytest.mark.parametrize("suffix", [".zip", ".tar"])
def test_s3_archive_with_stem_folder(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Members stored as ``Task04/...`` inside ``Task04.tar`` keep one stem.

    The Medical Segmentation Decathlon tars use this layout.
    """
    stem = "Task04_Hippocampus"
    instance_id = f"{stem}/imagesTr/hippocampus_001.nii.gz"
    payload = b"hippocampus"
    bucket_root = tmp_path / "bucket"
    bucket_root.mkdir()
    archive_path = bucket_root / f"{stem}{suffix}"
    members = {
        f"{stem}/imagesTr/._hippocampus_001.nii.gz": b"apple-double",
        instance_id: payload,
    }
    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
    else:
        with tarfile.open(archive_path, "w") as archive:
            for name, data in members.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    monkeypatch.setattr(_FakeS3, "root", bucket_root, raising=False)
    monkeypatch.setattr(_FakeS3, "bucket", _S3_BUCKET, raising=False)
    monkeypatch.setattr(
        "imgnet.download.downloaders.s3fs.S3FileSystem", _FakeS3
    )
    original_s3 = get_filesystem_class("s3")
    register_implementation("s3", _FakeS3, clobber=True)
    try:
        downloader = S3Downloader(_S3_BUCKET)
        assert instance_id in downloader.members
        output = tmp_path / "test_dir"
        downloader.download(output, instance_ids=[instance_id])
    finally:
        register_implementation("s3", original_s3, clobber=True)

    downloaded = output / instance_id
    assert downloaded.is_file(), sorted(
        p.relative_to(output).as_posix() for p in output.rglob("*")
    )
    assert downloaded.read_bytes() == payload
