"""Configuration models for building nnU-Net v2 raw datasets.

The configuration is a JSON file validated by [`NnUNetConfig`][imgnet.nnunet.config.NnUNetConfig].
Relative paths in the file are resolved against the file's own directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SUPPORTED_FILE_ENDINGS = (".nii.gz", ".nii", ".mha", ".nrrd")
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(ValueError):
    """Raised when an nnU-Net config file is missing or invalid."""


class Regex(BaseModel):
    """Match column values with a regular expression (``re.search``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regex: str = Field(
        description="Pattern searched for anywhere in the value.",
    )

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as e:
            msg = f"invalid regex {value!r}: {e}"
            raise ValueError(msg) from e
        return value


Scalar = Union[str, int, float, bool]
SelectorValue = Union[Scalar, list[Scalar], Regex]
Selector = dict[str, SelectorValue]
"""Column name -> value, list of allowed values, or ``{"regex": ...}``.

Every entry must match (AND). Values are compared as text, as they appear
in the query results CSV.
"""


class SplitColumn(BaseModel):
    """Use a collection's own train/test split column instead of a random split."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column holding the split value.")
    train: list[str] = Field(
        description="Values that put a sample in the training set.",
    )
    test: list[str] = Field(
        default_factory=list,
        description="Values that put a sample in the test set.",
    )

    @model_validator(mode="after")
    def _check_values(self) -> SplitColumn:
        if not self.train:
            msg = "split.train must list at least one value"
            raise ValueError(msg)
        overlap = set(self.train) & set(self.test)
        if overlap:
            msg = f"split values listed as both train and test: {sorted(overlap)}"
            raise ValueError(msg)
        return self


class CollectionConfig(BaseModel):
    """How to turn one collection's samples into nnU-Net cases."""

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(description="Collection name, as in the CSV.")
    method: Literal["combine", "remap"] = Field(
        description=(
            "'combine': merge binary ROI masks into one label map. "
            "'remap': map the values of one multi-label mask."
        ),
    )
    intensity_map: dict[str, int] = Field(
        description=(
            "combine: ROI name -> output label. "
            "remap: source mask value -> output label."
        ),
    )
    selector: Selector = Field(
        default_factory=dict,
        description="Row filter applied to both scans and masks.",
    )
    scan_selector: Selector = Field(
        default_factory=dict,
        description="Extra row filter applied to scans only.",
    )
    mask_selector: Selector = Field(
        default_factory=dict,
        description="Extra row filter applied to masks only.",
    )
    roi_column: str = Field(
        default="ROINames",
        description="combine: column holding the ROI name of each mask.",
    )
    allow_missing_rois: bool = Field(
        default=False,
        description=(
            "combine: keep samples that lack some configured ROIs. Off by "
            "default, because a missing mask file would be written as "
            "'no lesion' in the label."
        ),
    )
    group_column: str | None = Field(
        default=None,
        description=(
            "Patient ID column. Samples sharing a value are kept on the "
            "same side of a random split."
        ),
    )
    split: SplitColumn | None = Field(
        default=None,
        description=(
            "Use the collection's own split column. Falls back to a random "
            "split, with a warning, if the column has no values."
        ),
    )

    @model_validator(mode="after")
    def _check_intensity_map(self) -> CollectionConfig:
        if not self.intensity_map:
            msg = f"{self.collection}: intensity_map is empty"
            raise ValueError(msg)
        if self.method == "remap":
            bad = [k for k in self.intensity_map if not _is_int(k)]
            if bad:
                msg = (
                    f"{self.collection}: remap intensity_map keys must be "
                    f"integer mask values, got {bad}"
                )
                raise ValueError(msg)
            negative = [v for v in self.intensity_map.values() if v < 0]
        else:
            negative = [v for v in self.intensity_map.values() if v < 1]
        if negative:
            lowest = 0 if self.method == "remap" else 1
            msg = (
                f"{self.collection}: {self.method} output labels must be "
                f">= {lowest}, got {negative}"
            )
            raise ValueError(msg)
        return self

    def selector_columns(self) -> set[str]:
        """Columns this config reads, other than the split column."""
        cols = {
            *self.selector,
            *self.scan_selector,
            *self.mask_selector,
        }
        if self.method == "combine":
            cols.add(self.roi_column)
        if self.group_column is not None:
            cols.add(self.group_column)
        return cols


class NnUNetConfig(BaseModel):
    """Configuration for ``imgnet nnunet``."""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path = Field(
        description=(
            "Downloaded data root (the 'srcdata' folder). Files are read "
            "from <input_dir>/<Collection>/<filepath>."
        ),
    )
    query_results_csv: Path = Field(
        description="query_results.csv written by 'imgnet query'.",
    )
    nnunet_raw_dir: Path = Field(
        description="nnUNet_raw folder; the dataset folder is created here.",
    )
    dataset_id: int = Field(ge=1, le=999, description="nnU-Net dataset ID.")
    dataset_name: str = Field(
        description="Letters, digits, '_' and '-' only.",
    )
    labels: dict[str, int] = Field(
        description=(
            "Label name -> value. Must contain 'background': 0 and use "
            "consecutive values."
        ),
    )
    collections: list[CollectionConfig] = Field(min_length=1)
    channel_name: str = Field(
        default="CT",
        description="Input channel name written to channel_names['0'].",
    )
    file_ending: Literal[".nii.gz", ".nii", ".mha", ".nrrd"] = ".nii.gz"
    test_size: float = Field(default=0.2, ge=0.0, lt=1.0)
    random_state: int = 42
    prefix_with_collection: bool = Field(
        default=True,
        description="Case IDs are '<Collection>_<SampleID>' when true.",
    )
    scan_link_mode: Literal["copy", "symlink"] = Field(
        default="copy",
        description=(
            "'symlink' links scans instead of copying them when the source "
            "already has file_ending. Other scans are always converted."
        ),
    )
    geometry_tolerance: float = Field(
        default=1e-3,
        gt=0,
        description=(
            "Max allowed spacing/origin/direction difference between a mask "
            "and its scan. Smaller differences are snapped to the scan."
        ),
    )
    num_workers: int | None = Field(
        default=None,
        ge=1,
        description="Worker processes. Defaults to the available CPUs.",
    )

    @field_validator("dataset_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _DATASET_NAME_RE.fullmatch(value):
            msg = (
                f"dataset_name {value!r} may only contain letters, digits, "
                "'_' and '-'"
            )
            raise ValueError(msg)
        return value

    @field_validator("labels")
    @classmethod
    def _check_labels(cls, value: dict[str, int]) -> dict[str, int]:
        if value.get("background") != 0:
            msg = "labels must contain 'background': 0"
            raise ValueError(msg)
        values = sorted(value.values())
        if values != list(range(len(values))):
            msg = (
                "label values must be unique and consecutive from 0, "
                f"got {values}"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_collections(self) -> NnUNetConfig:
        names = [c.collection for c in self.collections]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            msg = f"collections listed more than once: {dupes}"
            raise ValueError(msg)
        valid = set(self.labels.values())
        for coll in self.collections:
            unknown = sorted(set(coll.intensity_map.values()) - valid)
            if unknown:
                msg = (
                    f"{coll.collection}: intensity_map uses label values "
                    f"{unknown} that are not in labels"
                )
                raise ValueError(msg)
        return self

    @property
    def dataset_folder_name(self) -> str:
        """Folder name nnU-Net expects, e.g. ``Dataset001_Lesions``."""
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def dataset_dir(self) -> Path:
        """Full path of the dataset folder inside ``nnunet_raw_dir``."""
        return self.nnunet_raw_dir / self.dataset_folder_name


def load_config(path: Path | str) -> NnUNetConfig:
    """Read and validate a config file.

    Relative paths are resolved against the config file's directory.

    Raises
    ------
    ConfigError
        If the file can't be read or doesn't validate.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        msg = f"Could not read config {path}: {e}"
        raise ConfigError(msg) from e
    try:
        config = NnUNetConfig.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid config {path}:\n{e}"
        raise ConfigError(msg) from e

    base = path.resolve().parent
    for field in ("input_dir", "query_results_csv", "nnunet_raw_dir"):
        value = getattr(config, field).expanduser()
        if not value.is_absolute():
            value = base / value
        setattr(config, field, value)
    return config


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True
