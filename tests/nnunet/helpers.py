"""Tiny synthetic NIfTI collections for nnU-Net builder tests.

Volumes are 4x6x6 voxels so the whole suite runs in a few seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from imgnet.nnunet.dataset_config import NnUNetDatasetConfig

SHAPE = (4, 6, 6)  # z, y, x
SPACING = (0.8, 0.8, 2.0)
ORIGIN = (-10.0, -20.0, 5.0)


def write_volume(
    path: Path,
    arr: np.ndarray,
    spacing: tuple[float, ...] = SPACING,
    origin: tuple[float, ...] = ORIGIN,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def scan_array(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, SHAPE).astype(np.int16)


@dataclass
class Source:
    """A fake download folder plus its query results CSV."""

    root: Path
    srcdata: Path
    csv: Path
    rows: list[dict]

    def save_csv(self) -> None:
        pd.DataFrame(self.rows).to_csv(self.csv)  # index, like imgnet query

    def add(self, collection: str, sample_id: str, file_type: str,
            filepath: str, **extra: str) -> None:  # fmt: skip
        self.rows.append({
            "Collection": collection,
            "SampleID": sample_id,
            "file_type": file_type,
            "filepath": filepath,
            **extra,
        })


ATLAS = "Abdomen Atlas 3"
MSWAL = "MSWAL"


def liver_mask() -> np.ndarray:
    arr = np.zeros(SHAPE, np.uint8)
    arr[1, :2, :2] = 1
    return arr


def kidney_mask() -> np.ndarray:
    # Overlaps the liver mask at voxel (1, 1, 1).
    arr = np.zeros(SHAPE, np.uint8)
    arr[1:3, 1:3, 1:3] = 1
    return arr


def multilabel_mask() -> np.ndarray:
    arr = np.zeros(SHAPE, np.uint8)
    arr[1] = 1
    arr[2] = 2
    arr[3, 0, 0] = 9  # not in the intensity map
    return arr


def make_source(root: Path) -> Source:
    """Two collections.

    Abdomen Atlas 3 (combine): A1-A4; A4 has no kidney_lesion mask.
    Patients: P1 = A1+A2, P2 = A3, P3 = A4.
    MSWAL (remap): M1, M2 (Split=Tr), M3 (Split=Ts); M2's scan is .nii.
    """
    srcdata = root / "srcdata"
    src = Source(root, srcdata, root / "query_results.csv", [])

    patients = {"A1": "P1", "A2": "P1", "A3": "P2", "A4": "P3"}
    for i, (sid, patient) in enumerate(patients.items()):
        base = f"images/{sid}"
        write_volume(srcdata / ATLAS / base / "ct.nii.gz", scan_array(i))
        src.add(ATLAS, sid, "scan", f"{base}/ct.nii.gz",
                PatientID=patient)  # fmt: skip
        rois = {"liver_lesion": liver_mask(), "aorta": liver_mask()}
        if sid != "A4":
            rois["kidney_lesion"] = kidney_mask()
        for roi, arr in rois.items():
            rel = f"masks/{sid}/{roi}.nii.gz"
            write_volume(srcdata / ATLAS / rel, arr)
            src.add(ATLAS, sid, "mask", rel, ROINames=roi,
                    PatientID=patient)  # fmt: skip

    for i, (sid, split) in enumerate({"M1": "Tr", "M2": "Tr", "M3": "Ts"}.items()):
        ext = ".nii" if sid == "M2" else ".nii.gz"
        scan_rel = f"imagesTr/{sid}_0000{ext}"
        write_volume(srcdata / MSWAL / scan_rel, scan_array(10 + i))
        src.add(MSWAL, sid, "scan", scan_rel, Split=split)
        mask_rel = f"labelsTr/{sid}.nii.gz"
        write_volume(srcdata / MSWAL / mask_rel, multilabel_mask())
        src.add(MSWAL, sid, "mask", mask_rel, Split=split)

    src.save_csv()
    return src


def dataset_config_dict(src: Source, **overrides: object) -> dict:
    config: dict = {
        "input_dir": str(src.srcdata),
        "query_results_csv": str(src.csv),
        "nnunet_raw_dir": str(src.root / "nnUNet_raw"),
        "dataset_id": 7,
        "dataset_name": "Lesions",
        "labels": {"background": 0, "liver tumor": 1, "kidney tumor": 2},
        "test_size": 0.34,
        "num_workers": 1,
        "collections": [
            {
                "collection": ATLAS,
                "method": "combine",
                "intensity_map": {"liver_lesion": 1, "kidney_lesion": 2},
                "group_column": "PatientID",
            },
            {
                "collection": MSWAL,
                "method": "remap",
                "intensity_map": {"1": 1, "2": 2},
                "split": {"column": "Split", "train": ["Tr"], "test": ["Ts"]},
            },
        ],
    }
    config.update(overrides)
    return config


def make_dataset_config(src: Source, **overrides: object) -> NnUNetDatasetConfig:
    return NnUNetDatasetConfig.model_validate(dataset_config_dict(src, **overrides))


def write_dataset_config(src: Source, **overrides: object) -> Path:
    path = src.root / "config.json"
    path.write_text(json.dumps(dataset_config_dict(src, **overrides)))
    return path


@pytest.fixture
def source(tmp_path: Path) -> Source:
    return make_source(tmp_path)
