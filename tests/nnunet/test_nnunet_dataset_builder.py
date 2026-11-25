"""End-to-end builds on tiny synthetic volumes."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from click.testing import CliRunner

from imgnet.cli.__main__ import cli
from imgnet.nnunet import dataset_builder
from imgnet.nnunet.case_selection import CaseSelectionError
from imgnet.nnunet.case_writer import CaseJob, WriteOptions, image_path
from imgnet.nnunet.dataset_builder import (
    MANIFEST_NAME,
    WORKER_DIED,
    build_nnunet_dataset,
    verify_dataset_files,
)
from imgnet.nnunet.errors import DatasetExistsError, DatasetOutputError
from tests.nnunet.helpers import (
    ATLAS,
    MSWAL,
    ORIGIN,
    SPACING,
    Source,
    kidney_mask,
    liver_mask,
    make_dataset_config,
    multilabel_mask,
    read_array,
    write_dataset_config,
    write_volume,
)

DATASET = "Dataset007_Lesions"


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_build_writes_valid_nnunet_layout(source: Source) -> None:
    result = build_nnunet_dataset(make_dataset_config(source))
    out = source.root / "nnUNet_raw" / DATASET
    assert result.dataset_dir == out
    assert not result.failures

    tr_images = _files(out / "imagesTr")
    tr_labels = _files(out / "labelsTr")
    ts_images = _files(out / "imagesTs")
    ts_labels = _files(out / "labelsTs")
    # Every image has its label, with nnU-Net's naming.
    assert [n.replace("_0000", "") for n in tr_images] == tr_labels
    assert [n.replace("_0000", "") for n in ts_images] == ts_labels
    assert "MSWAL_M3.nii.gz" in ts_labels  # from the Split column
    assert {"MSWAL_M1.nii.gz", "MSWAL_M2.nii.gz"} <= set(tr_labels)
    # A4 lacks kidney_lesion, so it's skipped entirely.
    assert not any("A4" in n for n in tr_labels + ts_labels)

    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset == {
        "name": DATASET,
        "description": f"{DATASET}, built by imgnet from {ATLAS}, {MSWAL}",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver tumor": 1, "kidney tumor": 2},
        "numTraining": len(tr_labels),
        "file_ending": ".nii.gz",
        "numTest": len(ts_labels),
    }
    assert len(tr_labels) + len(ts_labels) == 6  # A1-A3, M1-M3

    manifest = pd.read_csv(out / MANIFEST_NAME)
    status = dict(zip(manifest["sample_id"], manifest["status"]))
    assert status == {s: "written" for s in ["A1", "A2", "A3", "M1", "M2", "M3"]} | {
        "A4": "skipped"
    }


def test_labels_are_combined_and_remapped(source: Source) -> None:
    build_nnunet_dataset(make_dataset_config(source))
    out = source.root / "nnUNet_raw" / DATASET
    # Every Atlas sample has the same masks, so any of its labels will do.
    atlas = next(out.glob("labels*/AbdomenAtlas3_*"))

    expected = np.zeros_like(liver_mask())
    expected[liver_mask() > 0] = 1
    expected[kidney_mask() > 0] = 2  # listed later, wins the overlap
    assert np.array_equal(read_array(atlas), expected)

    remapped = read_array(out / "labelsTr" / "MSWAL_M1.nii.gz")
    source_mask = multilabel_mask()
    assert np.array_equal(remapped, np.where(source_mask == 9, 0, source_mask))
    assert remapped.dtype == np.uint8


def test_scans_are_copied_or_converted(source: Source) -> None:
    build_nnunet_dataset(make_dataset_config(source))
    out = source.root / "nnUNet_raw" / DATASET / "imagesTr"
    copied = out / "MSWAL_M1_0000.nii.gz"
    original = source.srcdata / MSWAL / "imagesTr/M1_0000.nii.gz"
    assert copied.read_bytes() == original.read_bytes()
    # M2's source is .nii; it must be converted, not just renamed.
    converted = sitk.ReadImage(str(out / "MSWAL_M2_0000.nii.gz"))
    assert converted.GetSpacing() == pytest.approx(SPACING)
    src_nii = source.srcdata / MSWAL / "imagesTr/M2_0000.nii"
    assert np.array_equal(sitk.GetArrayFromImage(converted), read_array(src_nii))


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin")
def test_symlink_mode(source: Source) -> None:
    build_nnunet_dataset(make_dataset_config(source, scan_link_mode="symlink"))
    out = source.root / "nnUNet_raw" / DATASET / "imagesTr"
    linked = out / "MSWAL_M1_0000.nii.gz"
    assert linked.is_symlink()
    assert linked.resolve() == (
        source.srcdata / MSWAL / "imagesTr/M1_0000.nii.gz"
    ).resolve()
    # .nii sources still have to be converted.
    assert not (out / "MSWAL_M2_0000.nii.gz").is_symlink()


def _write_mask_with_geometry(
    path: Path, spacing: tuple, origin: tuple, direction: tuple
) -> None:
    image = sitk.GetImageFromArray(multilabel_mask())
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_label_geometry_noise_is_snapped_to_scan(source: Source) -> None:
    noisy_direction = (1.0, 2e-4, 0.0, -2e-4, 1.0, 0.0, 0.0, 0.0, 1.0)
    _write_mask_with_geometry(
        source.srcdata / MSWAL / "labelsTr/M1.nii.gz",
        spacing=(SPACING[0] + 1e-5, SPACING[1], SPACING[2] - 1e-5),
        origin=(ORIGIN[0] + 4e-4, ORIGIN[1] - 4e-4, ORIGIN[2]),
        direction=noisy_direction,
    )
    result = build_nnunet_dataset(make_dataset_config(source))
    assert not result.failures
    label = sitk.ReadImage(
        str(source.root / "nnUNet_raw" / DATASET / "labelsTr/MSWAL_M1.nii.gz")
    )
    scan = sitk.ReadImage(
        str(source.srcdata / MSWAL / "imagesTr/M1_0000.nii.gz")
    )
    assert label.GetSpacing() == scan.GetSpacing()
    assert label.GetOrigin() == scan.GetOrigin()
    assert label.GetDirection() == scan.GetDirection() == IDENTITY


def test_direction_mismatch_fails_the_case(source: Source) -> None:
    flipped = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    _write_mask_with_geometry(
        source.srcdata / MSWAL / "labelsTr/M1.nii.gz",
        spacing=SPACING,
        origin=ORIGIN,
        direction=flipped,
    )
    result = build_nnunet_dataset(make_dataset_config(source))
    errors = {f.case.case_id: f.error for f in result.failures}
    assert list(errors) == ["MSWAL_M1"]
    assert "direction" in errors["MSWAL_M1"]


def test_misaligned_masks_fail_cleanly(source: Source) -> None:
    shifted = (ORIGIN[0] + 5.0, ORIGIN[1], ORIGIN[2])
    write_volume(
        source.srcdata / MSWAL / "labelsTr/M1.nii.gz",
        multilabel_mask(),
        origin=shifted,
    )
    bigger = np.zeros((5, 6, 6), np.uint8)
    write_volume(source.srcdata / MSWAL / "labelsTr/M2.nii.gz", bigger)

    result = build_nnunet_dataset(make_dataset_config(source))
    errors = {f.case.case_id: f.error for f in result.failures}
    assert set(errors) == {"MSWAL_M1", "MSWAL_M2"}
    assert "origin" in errors["MSWAL_M1"]
    assert "size" in errors["MSWAL_M2"]

    out = source.root / "nnUNet_raw" / DATASET
    leftovers = [p for p in out.rglob("MSWAL_M[12]*")]
    assert leftovers == []
    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["numTraining"] == len(_files(out / "imagesTr"))
    manifest = pd.read_csv(out / MANIFEST_NAME).set_index("sample_id")
    assert manifest.loc["M1", "status"] == "failed"


def test_existing_output_is_protected(source: Source) -> None:
    config = make_dataset_config(source)
    build_nnunet_dataset(config)
    with pytest.raises(DatasetExistsError, match="--overwrite"):
        build_nnunet_dataset(config)

    stray = config.dataset_dir / "imagesTr" / "stray_0000.nii.gz"
    stray.write_bytes(b"")
    build_nnunet_dataset(config, overwrite=True)
    assert not stray.exists()


def test_dataset_id_must_be_unique(source: Source) -> None:
    (source.root / "nnUNet_raw" / "Dataset007_Other").mkdir(parents=True)
    with pytest.raises(DatasetExistsError, match="already used"):
        build_nnunet_dataset(make_dataset_config(source))


def test_dry_run_writes_nothing(source: Source) -> None:
    result = build_nnunet_dataset(make_dataset_config(source), dry_run=True)
    assert (result.num_train, result.num_test) == (
        sum(c.split == "Tr" for c in result.selection.cases),
        sum(c.split == "Ts" for c in result.selection.cases),
    )
    assert result.num_train + result.num_test == 6
    assert not (source.root / "nnUNet_raw").exists()


def test_query_results_without_masks_are_an_error(source: Source) -> None:
    for row in source.rows:
        row["file_type"] = "scan" if row["file_type"] == "scan" else "other"
    source.save_csv()
    with pytest.raises(CaseSelectionError, match="have no masks"):
        build_nnunet_dataset(make_dataset_config(source))
    collections = [make_dataset_config(source).model_dump()["collections"][1]]
    with pytest.raises(CaseSelectionError, match="MSWAL: no cases selected"):
        build_nnunet_dataset(make_dataset_config(source, collections=collections))


def test_worker_pool_matches_serial(source: Source, tmp_path: Path) -> None:
    serial = build_nnunet_dataset(make_dataset_config(source, num_workers=1))
    pooled = build_nnunet_dataset(
        make_dataset_config(source, nnunet_raw_dir=str(tmp_path / "pooled")),
        num_workers=2,
    )
    assert not pooled.failures
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        names = _files(serial.dataset_dir / sub)
        assert names == _files(pooled.dataset_dir / sub)
        for name in names:
            a = read_array(serial.dataset_dir / sub / name)
            b = read_array(pooled.dataset_dir / sub / name)
            assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dry_run_and_build(source: Source) -> None:
    config_path = write_dataset_config(source)
    runner = CliRunner()

    result = runner.invoke(cli, ["nnunet", str(config_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (source.root / "nnUNet_raw").exists()

    result = runner.invoke(cli, ["nnunet", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (source.root / "nnUNet_raw" / DATASET / "dataset.json").exists()

    result = runner.invoke(cli, ["nnunet", str(config_path)])
    assert result.exit_code == 1
    assert "--overwrite" in result.output


def test_cli_reports_config_errors(source: Source) -> None:
    config_path = write_dataset_config(source, dataset_name="has space")
    result = CliRunner().invoke(cli, ["nnunet", str(config_path)])
    assert result.exit_code == 1
    assert "dataset_name" in result.output


def test_cli_exits_nonzero_when_cases_fail(source: Source) -> None:
    write_volume(
        source.srcdata / MSWAL / "labelsTr/M1.nii.gz",
        multilabel_mask(),
        origin=(99.0, 0.0, 0.0),
    )
    result = CliRunner().invoke(cli, ["nnunet", str(write_dataset_config(source))])
    assert result.exit_code == 1
    assert "failed=1" in result.output


# ---------------------------------------------------------------------------
# Dry run, cleanup, verification, reporting
# ---------------------------------------------------------------------------


def test_dry_run_runs_the_output_checks(source: Source) -> None:
    config = make_dataset_config(source)
    build_nnunet_dataset(config)
    with pytest.raises(DatasetExistsError, match="--overwrite"):
        build_nnunet_dataset(config, dry_run=True)
    build_nnunet_dataset(config, dry_run=True, overwrite=True)  # allowed
    assert (config.dataset_dir / "dataset.json").exists()  # not deleted

    (source.root / "nnUNet_raw" / "Dataset007_Other").mkdir()
    with pytest.raises(DatasetExistsError, match="already used"):
        build_nnunet_dataset(config, dry_run=True, overwrite=True)


def test_parent_cleans_up_after_a_dead_worker(
    source: Source,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A worker killed mid-case (e.g. OOM) can't clean up after itself."""
    real_attempt = dataset_builder._attempt

    def dies_after_writing_scan(job: CaseJob, options: WriteOptions):  # noqa: ANN202
        # Simulate the parent's view of a worker that died mid-case: a
        # partial scan on disk and a WORKER_DIED outcome.
        if job.case_id.startswith("MSWAL"):
            dst = image_path(
                options.dataset_dir, job.case_id, job.split, options.file_ending
            )
            dst.write_bytes(b"partial")
            return WORKER_DIED
        return real_attempt(job, options)

    monkeypatch.setattr(dataset_builder, "_attempt", dies_after_writing_scan)
    with caplog.at_level(logging.ERROR, logger="imgnet"):
        result = build_nnunet_dataset(make_dataset_config(source))

    assert {f.case.case_id for f in result.failures} == {
        "MSWAL_M1", "MSWAL_M2", "MSWAL_M3",
    }  # fmt: skip
    out = source.root / "nnUNet_raw" / DATASET
    assert list(out.rglob("MSWAL_*")) == []  # partial files removed
    # One summary line for the dead worker, not one per case.
    assert caplog.text.count("worker process died") == 1
    assert "Failed MSWAL" not in caplog.text
    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["numTraining"] == len(_files(out / "imagesTr"))


def test_verification_catches_stray_and_missing_files(source: Source) -> None:
    config = make_dataset_config(source)
    result = build_nnunet_dataset(config)
    verify_dataset_files(config, result.written)  # clean build passes

    stray = config.dataset_dir / "imagesTr" / "Leftover_0000.nii.gz"
    stray.write_bytes(b"")
    with pytest.raises(DatasetOutputError, match="unexpected files"):
        verify_dataset_files(config, result.written)
    stray.unlink()

    (config.dataset_dir / "labelsTs").joinpath("MSWAL_M3.nii.gz").unlink()
    with pytest.raises(DatasetOutputError, match="labelsTs is missing"):
        verify_dataset_files(config, result.written)

    # Other files (e.g. .DS_Store) are ignored, as nnU-Net ignores them.
    (config.dataset_dir / "labelsTs" / "MSWAL_M3.nii.gz").write_bytes(b"")
    (config.dataset_dir / "imagesTr" / ".DS_Store").write_bytes(b"")
    verify_dataset_files(config, result.written)


def test_unmapped_values_are_reported_once_per_collection(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        build_nnunet_dataset(make_dataset_config(source))
    # Every MSWAL mask has value 9, which isn't mapped.
    assert caplog.text.count("not in intensity_map") == 1
    assert "MSWAL: 3 cases have mask values" in caplog.text
    assert "9 (3 cases)" in caplog.text

    manifest = pd.read_csv(
        source.root / "nnUNet_raw" / DATASET / MANIFEST_NAME
    ).set_index("sample_id")
    assert manifest.loc["M1", "notes"] == (
        "mask values not in intensity_map set to 0: [9]"
    )
    assert pd.isna(manifest.loc["A1", "notes"])


def test_workers_do_not_import_pandas_or_logging() -> None:
    """Spawned workers only need case_writer; it must stay lightweight."""
    code = (
        "import sys, imgnet.nnunet.case_writer; "
        "print([m for m in ('pandas', 'pyarrow', 'structlog', "
        "'imgnet.loggers', 'imgnet.nnunet.case_selection') "
        "if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "[]"


def test_cli_reports_errors_without_tracebacks(source: Source) -> None:
    runner = CliRunner()
    config_path = write_dataset_config(source, random_state=-1)
    result = runner.invoke(cli, ["nnunet", str(config_path)])
    assert result.exit_code == 1
    assert "random_state" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)

    config_path = write_dataset_config(
        source, input_dir=str(source.root / "missing")
    )
    result = runner.invoke(cli, ["nnunet", str(config_path)])
    assert result.exit_code == 1
    assert "input_dir does not exist" in result.output
    assert isinstance(result.exception, SystemExit)
