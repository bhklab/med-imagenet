from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class FileType(Enum):
    DICOM = "dicom"
    NIFTI = "nifti"


class TCIASource(BaseModel):
    file_type: FileType = FileType.DICOM
    source: Literal["tcia"] = "tcia"
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])

class PrivateTCIASource(BaseModel):
    file_type: FileType = FileType.DICOM
    source: Literal["private_tcia"] = "private_tcia"
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])


class DropboxSource(BaseModel):
    file_type: FileType
    source: Literal["dropbox"] = "dropbox"
    url: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")


class S3Source(BaseModel):
    file_type: FileType
    source: Literal["s3"] = "s3"
    bucket_name: str
    filenames: list[str] | None = None
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")


class ZenodoSource(BaseModel):
    file_type: FileType
    source: Literal["zenodo"] = "zenodo"
    record_id: str
    filenames: list[str] | None = None
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")


class HuggingFaceSource(BaseModel):
    file_type: FileType
    source: Literal["huggingface"] = "huggingface"
    repo_id: str
    post_download: list[str] = Field(default_factory=lambda: ["unzip"])
    description: str = Field(default="")


SourceConfig = Annotated[
    TCIASource | DropboxSource | S3Source | ZenodoSource | HuggingFaceSource,
    Field(discriminator="source"),
]

source_adapter: TypeAdapter[SourceConfig] = TypeAdapter(SourceConfig)
