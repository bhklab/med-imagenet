"""Image I/O for one nnU-Net case. Runs inside worker processes.

Kept free of pandas and imgnet logging so worker start-up stays cheap;
warnings are returned to the parent process instead of logged here.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from pathlib import Path

    from imgnet.nnunet.case_selection import SelectedCase


class GeometryError(ValueError):
    """A mask is not on the same voxel grid as its scan."""


@dataclass(frozen=True)
class WriteOptions:
    """Settings shared by every case of a build."""

    dataset_dir: Path
    file_ending: str
    scan_link_mode: str
    geometry_tolerance: float


@dataclass
class CaseWriteResult:
    """Outcome of writing one case."""

    case_id: str
    split: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Grid:
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


def init_worker() -> None:
    """Limit ITK to one thread; parallelism comes from the process pool."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def image_path(
    dataset_dir: Path, case: SelectedCase, file_ending: str
) -> Path:
    """Destination of the case's scan (channel 0)."""
    name = f"{case.case_id}_0000{file_ending}"
    return dataset_dir / f"images{case.split}" / name


def label_path(
    dataset_dir: Path, case: SelectedCase, file_ending: str
) -> Path:
    """Destination of the case's label map."""
    return dataset_dir / f"labels{case.split}" / f"{case.case_id}{file_ending}"


def write_case(case: SelectedCase, options: WriteOptions) -> CaseWriteResult:
    """Build the label, then write scan and label for one case.

    On any error both outputs are removed, so a failed case never leaves a
    half-written pair behind.
    """
    img_dst = image_path(options.dataset_dir, case, options.file_ending)
    lbl_dst = label_path(options.dataset_dir, case, options.file_ending)
    result = CaseWriteResult(case.case_id, case.split)
    try:
        grid = _read_grid(case.scan)
        label = build_label(case, grid, options.geometry_tolerance, result)
        _place_scan(case.scan, img_dst, options)
        sitk.WriteImage(label, str(lbl_dst), useCompression=True)
    except BaseException:
        img_dst.unlink(missing_ok=True)
        lbl_dst.unlink(missing_ok=True)
        raise
    return result


def build_label(
    case: SelectedCase, grid: _Grid, tolerance: float, result: CaseWriteResult
) -> sitk.Image:
    """Combine or remap the case's masks into one label map on *grid*."""
    out: np.ndarray | None = None
    if case.method == "combine":
        top = max(case.labels)
        for mask_path, value in zip(case.masks, case.labels, strict=True):
            arr = _read_mask(mask_path, grid, tolerance)
            if out is None:
                out = np.zeros(arr.shape, dtype=_label_dtype(top))
            out[arr > 0] = value
    else:
        (mask_path,) = case.masks
        arr = _read_mask(mask_path, grid, tolerance)
        out = np.zeros(arr.shape, dtype=_label_dtype(max(case.remap.values())))
        mapped = np.zeros(arr.shape, dtype=bool)
        for source, target in case.remap.items():
            hit = arr == source
            out[hit] = target
            mapped |= hit
        unmapped = np.unique(arr[~mapped & (arr != 0)])
        if unmapped.size:
            result.warnings.append(
                f"{case.case_id}: mask values {unmapped.tolist()} are not in "
                "intensity_map and were set to 0"
            )
    assert out is not None
    label = sitk.GetImageFromArray(out)
    label.SetSpacing(grid.spacing)
    label.SetOrigin(grid.origin)
    label.SetDirection(grid.direction)
    return label


def _label_dtype(max_value: int) -> type[np.unsignedinteger]:
    return np.uint8 if max_value <= np.iinfo(np.uint8).max else np.uint16


def _read_grid(path: Path) -> _Grid:
    """Read only the header of an image."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return _Grid(
        size=tuple(reader.GetSize()),
        spacing=tuple(reader.GetSpacing()),
        origin=tuple(reader.GetOrigin()),
        direction=tuple(reader.GetDirection()),
    )


def _read_mask(path: Path, grid: _Grid, tolerance: float) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    check_geometry(image, grid, tolerance, path)
    return sitk.GetArrayFromImage(image)


def check_geometry(
    image: sitk.Image, grid: _Grid, tolerance: float, path: Path
) -> None:
    """Raise if *image* isn't on *grid* within *tolerance*.

    Raises
    ------
    GeometryError
        If the size differs, or spacing, origin or direction differ by more
        than *tolerance*.
    """
    if tuple(image.GetSize()) != grid.size:
        msg = (
            f"{path.name}: size {image.GetSize()} does not match scan "
            f"{grid.size}"
        )
        raise GeometryError(msg)
    checks = (
        ("spacing", image.GetSpacing(), grid.spacing),
        ("origin", image.GetOrigin(), grid.origin),
        ("direction", image.GetDirection(), grid.direction),
    )
    for name, actual, expected in checks:
        if not np.allclose(actual, expected, rtol=0, atol=tolerance):
            msg = (
                f"{path.name}: {name} {np.round(actual, 4).tolist()} does "
                f"not match scan {np.round(expected, 4).tolist()} "
                f"(tolerance {tolerance})"
            )
            raise GeometryError(msg)


def _place_scan(src: Path, dst: Path, options: WriteOptions) -> None:
    if not src.name.endswith(options.file_ending):
        sitk.WriteImage(
            sitk.ReadImage(str(src)), str(dst), useCompression=True
        )
    elif options.scan_link_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copyfile(src, dst)
