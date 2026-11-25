"""Image I/O for one nnU-Net case. Runs inside worker processes.

Workers receive a `CaseJob` and import only this module, numpy and
SimpleITK: no pandas and no imgnet logging setup. This keeps start-up cheap
and avoids every worker re-running the log-file setup. Anything worth
reporting is returned to the parent process in a `CaseWriteResult`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from pathlib import Path

_MAX_REPORTED_VALUES = 10


class GeometryError(ValueError):
    """A mask is not on the same voxel grid as its scan."""


@dataclass(frozen=True)
class WriteOptions:
    """Settings shared by every case of a build."""

    dataset_dir: Path
    file_ending: str
    scan_link_mode: str
    geometry_tolerance: float


@dataclass(frozen=True)
class CaseJob:
    """Everything a worker needs to write one case (cheap to pickle)."""

    case_id: str
    split: str
    method: str
    scan: Path
    masks: tuple[Path, ...]
    # combine: output label per mask, in order (later wins).
    labels: tuple[int, ...] = ()
    # remap: (source value, output label) pairs.
    remap: tuple[tuple[int, int], ...] = ()


@dataclass
class CaseWriteResult:
    """Outcome of writing one case."""

    case_id: str
    # remap: distinct non-zero mask values not in intensity_map (set to 0),
    # at most _MAX_REPORTED_VALUES of them, and how many there were.
    unmapped_values: list[float] = field(default_factory=list)
    num_unmapped_values: int = 0


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
    dataset_dir: Path, case_id: str, split: str, file_ending: str
) -> Path:
    """Destination of a case's scan (channel 0)."""
    return dataset_dir / f"images{split}" / f"{case_id}_0000{file_ending}"


def label_path(
    dataset_dir: Path, case_id: str, split: str, file_ending: str
) -> Path:
    """Destination of a case's label map."""
    return dataset_dir / f"labels{split}" / f"{case_id}{file_ending}"


def remove_case_outputs(job: CaseJob, options: WriteOptions) -> None:
    """Delete whatever exists of a case's two output files."""
    for path in (
        image_path(
            options.dataset_dir, job.case_id, job.split, options.file_ending
        ),
        label_path(
            options.dataset_dir, job.case_id, job.split, options.file_ending
        ),
    ):
        path.unlink(missing_ok=True)


def write_case(case: CaseJob, options: WriteOptions) -> CaseWriteResult:
    """Build the label, then write scan and label for one case.

    On a Python exception both outputs are removed here. If the worker
    process dies instead (for example out of memory), the parent removes
    them; see `remove_case_outputs`.
    """
    img_dst = image_path(
        options.dataset_dir, case.case_id, case.split, options.file_ending
    )
    lbl_dst = label_path(
        options.dataset_dir, case.case_id, case.split, options.file_ending
    )
    result = CaseWriteResult(case.case_id)
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
    case: CaseJob, grid: _Grid, tolerance: float, result: CaseWriteResult
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
        top = max(target for _, target in case.remap)
        out = np.zeros(arr.shape, dtype=_label_dtype(top))
        mapped = np.zeros(arr.shape, dtype=bool)
        for source, target in case.remap:
            hit = arr == source
            out[hit] = target
            mapped |= hit
        unmapped = np.unique(arr[~mapped & (arr != 0)])
        result.num_unmapped_values = int(unmapped.size)
        result.unmapped_values = [
            _plain_number(v) for v in unmapped[:_MAX_REPORTED_VALUES]
        ]
    assert out is not None
    label = sitk.GetImageFromArray(out)
    label.SetSpacing(grid.spacing)
    label.SetOrigin(grid.origin)
    label.SetDirection(grid.direction)
    return label


def _plain_number(value: np.generic) -> float:
    number = float(value)
    return int(number) if number.is_integer() else number


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
