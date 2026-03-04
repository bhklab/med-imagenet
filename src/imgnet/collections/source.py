from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class FileType(Enum):
    DICOM = "dicom"
    NIFTI = "nifti"


class TCIASource(BaseModel):
    file_type: FileType = FileType.DICOM
    source: Literal["tcia"] = "tcia"
    post_download: list[str] = []


class DropboxSource(BaseModel):
    file_type: FileType
    source: Literal["dropbox"] = "dropbox"
    url: str
    post_download: list[str] = []


class S3Source(BaseModel):
    file_type: FileType
    source: Literal["s3"] = "s3"
    bucket_name: str
    file_name: str
    post_download: list[str] = []


class ZenodoSource(BaseModel):
    file_type: FileType
    source: Literal["zenodo"] = "zenodo"
    record_id: str
    filename: str | None = None
    post_download: list[str] = []


SourceConfig = Annotated[
    TCIASource | DropboxSource | S3Source | ZenodoSource,
    Field(discriminator="source"),
]

source_adapter = TypeAdapter(SourceConfig)
