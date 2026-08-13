from enum import Enum
from typing import Annotated, Literal
from abc import abstractmethod
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from typing import List
from pathlib import Path
from pydantic import Field

from imgnet.download.base import BaseDownloader
from imgnet.download.downloaders import (
    DropboxDownloader,
    HuggingFaceDownloader,
    IDCDownloader,
    NBIADownloader,
    S3Downloader,
    ZenodoDownloader,
    LMUMunichDownloader,
    GoogleDriveDownloader,
    GitHubDownloader,
    CompositeDownloader
)


class FileType(Enum):
    DICOM = "dicom"
    NIFTI = "nifti"

class BaseSource(BaseModel):
    file_type: FileType
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    source: str

    @abstractmethod
    def get_downloader(self) -> BaseDownloader:
        """Get the downloader for this source."""
        raise NotImplementedError

class TCIASource(BaseSource):
    name: str
    file_type: FileType = FileType.DICOM
    source: Literal["tcia"] = "tcia"
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])

    def get_downloader(self)-> BaseDownloader:
        return IDCDownloader(self.name)


class PrivateTCIASource(BaseSource):
    name: str
    file_type: FileType = FileType.DICOM
    source: Literal["private_tcia"] = "private_tcia"
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])

    def get_downloader(self)-> BaseDownloader:
            return NBIADownloader(self.name)

class DropboxSource(BaseSource):
    file_type: FileType
    source: Literal["dropbox"] = "dropbox"
    url: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return DropboxDownloader(self.url)

class GoogleDriveSource(BaseSource):
    file_type: FileType
    source: Literal["google drive"] = "google drive"
    url: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return GoogleDriveDownloader(self.url)
    
class GitHubSource(BaseSource):
    file_type: FileType
    source: Literal["github"] = "github"
    repo_id: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return GitHubDownloader(self.repo_id)

class S3Source(BaseSource):
    file_type: FileType
    source: Literal["s3"] = "s3"
    bucket_name: str
    filenames: list[str] | None = None
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return S3Downloader(self.bucket_name)


class ZenodoSource(BaseSource):
    file_type: FileType
    source: Literal["zenodo"] = "zenodo"
    record_id: str
    filenames: list[str] | None = None
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return ZenodoDownloader(self.record_id)

class LMUMunichSource(BaseSource):
    file_type: FileType
    source: Literal["lmu munich"] = "lmu munich"
    record_id: str
    filenames: list[str] | None = None
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return LMUMunichDownloader(self.record_id)



class HuggingFaceSource(BaseSource):
    file_type: FileType
    source: Literal["huggingface"] = "huggingface"
    repo_id: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")

    def get_downloader(self)-> BaseDownloader:
            return HuggingFaceDownloader(self.repo_id)

class CompositeSource(BaseSource):
    """A source that combines one or more other sources."""
    source: Literal["composite"] = "composite"
    sources: List[BaseSource]  # List of any valid sources
    # Inherits file_type, post_download, description from BaseSource
    
    def get_downloader(self) -> BaseDownloader:
        """Return a special CompositeDownloader."""
        return CompositeDownloader([source.get_downloader() for source in self.sources])
    
    @model_validator(mode="after")
    def validate_composite(self) -> "CompositeSource":
        """Validate that all sources have the same file_type."""
        if not self.sources:
            raise ValueError("CompositeSource must have at least one source")
        
        # Check all sources have the same file_type
        first_file_type = self.sources[0].file_type
        for source in self.sources[1:]:
            if source.file_type != first_file_type:
                msg = f"All sources in composite must have the same file_type.\nFound '{first_file_type}' and '{source.file_type}' in source '{source.source}'"
                raise ValueError(
                    msg
                )
        
        # Ensure composite's file_type matches the sources
        if self.file_type != first_file_type:
            msg = f"Composite file_type '{self.file_type}' does not match source file_type '{first_file_type}'"
            raise ValueError(
                msg
            )
        
        return self


SourceConfig = Annotated[
    TCIASource | DropboxSource | S3Source | ZenodoSource | HuggingFaceSource | LMUMunichSource | GoogleDriveSource | GitHubSource | CompositeSource,
    Field(discriminator="source"),
]

source_adapter: TypeAdapter[SourceConfig] = TypeAdapter(SourceConfig)
