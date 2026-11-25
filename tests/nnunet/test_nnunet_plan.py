"""Planning: selectors, ambiguity, skips, splits and case IDs (no image I/O)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from imgnet.nnunet.config import NnUNetConfig, Regex
from imgnet.nnunet.plan import (
    PlanError,
    assign_splits,
    make_case_id,
    plan_dataset,
    read_query_results,
    select_rows,
)
from tests.nnunet.helpers import (
    ATLAS,
    MSWAL,
    Source,
    config_dict,
    liver_mask,
    make_config,
    write_volume,
)


def _plan(src: Source, **overrides: object):  # noqa: ANN202
    config = make_config(src, **overrides)
    return plan_dataset(read_query_results(src.csv), config)


def _collections(src: Source, *changes: dict) -> list[dict]:
    collections = config_dict(src)["collections"]
    for coll, change in zip(collections, changes):
        coll.update(change)
    return collections


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def test_select_rows_value_forms() -> None:
    df = pd.DataFrame(
        {
            "Phase": ["C+V", "C+A", None, "DWI"],
            "ImageID": ["1", "2", "3", "10"],
        }
    )
    assert select_rows(df, {}).tolist() == [True] * 4
    assert select_rows(df, {"Phase": "C+V"}).tolist() == [1, 0, 0, 0]
    assert select_rows(df, {"Phase": ["C+V", "DWI"]}).tolist() == [1, 0, 0, 1]
    # Numbers are compared as text, so 1 matches "1" but not "10".
    assert select_rows(df, {"ImageID": 1}).tolist() == [1, 0, 0, 0]
    # Regex is searched; missing values never match.
    regex = {"Phase": Regex(regex=r"^C\+")}
    assert select_rows(df, regex).tolist() == [1, 1, 0, 0]
    both = {"Phase": Regex(regex="C"), "ImageID": ["2", "10"]}
    assert select_rows(df, both).tolist() == [0, 1, 0, 0]


# ---------------------------------------------------------------------------
# Resolving cases
# ---------------------------------------------------------------------------


def test_plan_resolves_cases_and_skips(source: Source) -> None:
    plan = _plan(source)
    by_id = {c.case_id: c for c in plan.cases}
    assert sorted(by_id) == [
        "AbdomenAtlas3_A1",
        "AbdomenAtlas3_A2",
        "AbdomenAtlas3_A3",
        "MSWAL_M1",
        "MSWAL_M2",
        "MSWAL_M3",
    ]
    a1 = by_id["AbdomenAtlas3_A1"]
    assert a1.scan == source.srcdata / ATLAS / "images/A1/ct.nii.gz"
    assert [p.name for p in a1.masks] == [
        "liver_lesion.nii.gz",
        "kidney_lesion.nii.gz",
    ]
    assert a1.labels == [1, 2]
    assert by_id["MSWAL_M1"].remap == {1: 1, 2: 2}

    assert [(s.sample_id, s.reason) for s in plan.skipped] == [
        ("A4", "missing ROIs ['kidney_lesion']")
    ]


def test_allow_missing_rois_keeps_partial_samples(source: Source) -> None:
    collections = _collections(source, {"allow_missing_rois": True})
    plan = _plan(source, collections=collections)
    a4 = next(c for c in plan.cases if c.sample_id == "A4")
    assert a4.labels == [1]
    assert not plan.skipped


def test_ambiguous_scans_fail_until_selected(source: Source) -> None:
    # Give M1 a second scan in another phase, like LLD-MMRI.
    for row in source.rows:
        if row["Collection"] == MSWAL:
            row["Phase"] = "C+V"
    extra = "imagesTr/M1_arterial.nii.gz"
    write_volume(source.srcdata / MSWAL / extra, liver_mask())
    source.add(MSWAL, "M1", "scan", extra, Split="Tr", Phase="C+A")
    source.save_csv()

    with pytest.raises(PlanError, match=r"MSWAL: 1 samples match more than one scan"):
        _plan(source)

    collections = _collections(source, {}, {"selector": {"Phase": "C+V"}})
    plan = _plan(source, collections=collections)
    m1 = next(c for c in plan.cases if c.sample_id == "M1")
    assert m1.scan.name == "M1_0000.nii.gz"


def test_ambiguous_masks_fail_until_selected(source: Source) -> None:
    extra = "labelsTr/M1_anatomy.nii.gz"
    write_volume(source.srcdata / MSWAL / extra, liver_mask())
    source.add(MSWAL, "M1", "mask", extra, Split="Tr", mask_type="anatomy")
    source.save_csv()

    with pytest.raises(PlanError, match="more than one mask"):
        _plan(source)

    selector = {"mask_selector": {"filepath": {"regex": r"M\d\.nii"}}}
    collections = _collections(source, {}, selector)
    plan = _plan(source, collections=collections)
    m1 = next(c for c in plan.cases if c.sample_id == "M1")
    assert [p.name for p in m1.masks] == ["M1.nii.gz"]


def test_duplicate_roi_masks_are_ambiguous(source: Source) -> None:
    source.add(ATLAS, "A1", "mask", "masks/A1/aorta.nii.gz",
               ROINames="liver_lesion", PatientID="P1")  # fmt: skip
    source.save_csv()
    with pytest.raises(PlanError, match=r"A1 \(liver_lesion\)"):
        _plan(source)


def test_missing_files_are_skipped(source: Source) -> None:
    (source.srcdata / MSWAL / "labelsTr/M2.nii.gz").unlink()
    plan = _plan(source)
    assert "M2" not in {c.sample_id for c in plan.cases}
    reasons = {s.sample_id: s.reason for s in plan.skipped}
    assert reasons["M2"].startswith("file not found:")


def test_samples_without_scan_or_mask_are_skipped(source: Source) -> None:
    source.add(MSWAL, "M9", "scan", "imagesTr/M9.nii.gz", Split="Ts")
    source.add(MSWAL, "M8", "mask", "labelsTr/M8.nii.gz", Split="Ts")
    source.save_csv()
    reasons = {s.sample_id: s.reason for s in _plan(source).skipped}
    assert reasons["M9"] == "no mask matched"
    assert reasons["M8"] == "no scan matched"


def test_missing_columns_and_collections(source: Source) -> None:
    collections = _collections(source, {"scan_selector": {"Phase": "x"}})
    with pytest.raises(PlanError, match=r"columns \['Phase'\] are not in"):
        _plan(source, collections=collections)

    collections = _collections(source, {"collection": "Nope"})
    with pytest.raises(PlanError, match=r"\['Nope'\] are configured"):
        _plan(source, collections=collections)

    df = read_query_results(source.csv).drop(columns=["file_type"])
    with pytest.raises(PlanError, match="missing required columns"):
        plan_dataset(df, make_config(source))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_column_is_used(source: Source) -> None:
    splits = {c.sample_id: c.split for c in _plan(source).cases}
    assert (splits["M1"], splits["M2"], splits["M3"]) == ("Tr", "Tr", "Ts")


def test_split_column_unmapped_value_fails(source: Source) -> None:
    for row in source.rows:
        if row.get("Split") == "Ts":
            row["Split"] = "Va"
    source.save_csv()
    with pytest.raises(PlanError, match=r"values \['Va'\] are not listed"):
        _plan(source)


def test_split_column_conflict_fails(source: Source) -> None:
    for row in source.rows:
        if row["SampleID"] == "M1" and row["file_type"] == "mask":
            row["Split"] = "Ts"
    source.save_csv()
    with pytest.raises(PlanError, match="more than one 'Split' value"):
        _plan(source)


def test_missing_split_column_falls_back_to_random(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    split = {"column": "OfficialSplit", "train": ["Tr"]}
    collections = _collections(source, {"split": split})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        plan = _plan(source, collections=collections)
    atlas = [c for c in plan.cases if c.collection == ATLAS]
    assert {c.split for c in atlas} == {"Tr", "Ts"}
    assert "using a random split" in caplog.text


def test_split_column_warns_that_group_column_is_ignored(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    collections = _collections(source, {}, {"group_column": "SampleID"})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        plan = _plan(source, collections=collections)
    splits = {c.sample_id: c.split for c in plan.cases}
    assert splits["M3"] == "Ts"  # still from the Split column
    assert "group_column 'SampleID' only applies to random splits" in caplog.text


def test_split_fallback_uses_group_column(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    split = {"column": "OfficialSplit", "train": ["Tr"]}
    collections = _collections(source, {"split": split})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        plan = _plan(source, collections=collections)
    atlas = {c.sample_id: c.split for c in plan.cases if c.collection == ATLAS}
    assert atlas["A1"] == atlas["A2"]  # same patient P1
    assert "grouped by 'PatientID'" in caplog.text


def _split_frame(n_patients: int, per_patient: int) -> pd.DataFrame:
    rows = [
        {"SampleID": f"S{p}_{i}", "Patient": f"P{p}"}
        for p in range(n_patients)
        for i in range(per_patient)
    ]
    return pd.DataFrame(rows)


def _coll(**changes: object):  # noqa: ANN202
    config = NnUNetConfig.model_validate(
        {
            "input_dir": ".",
            "query_results_csv": "q.csv",
            "nnunet_raw_dir": ".",
            "dataset_id": 1,
            "dataset_name": "X",
            "labels": {"background": 0, "a": 1},
            "collections": [
                {
                    "collection": "C",
                    "method": "combine",
                    "intensity_map": {"a": 1},
                    **changes,
                }
            ],
        }
    )
    return config.collections[0]


def test_random_split_keeps_patients_together() -> None:
    df = _split_frame(n_patients=20, per_patient=3)
    sids = df["SampleID"].tolist()
    splits = assign_splits(
        df, sids, _coll(group_column="Patient"), test_size=0.25, random_state=1
    )
    by_patient = df.assign(split=df["SampleID"].map(splits)).groupby("Patient")
    assert (by_patient["split"].nunique() == 1).all()
    test_patients = sum(by_patient["split"].first() == "Ts")
    assert test_patients == 5  # ceil(0.25 * 20)


def test_random_split_is_deterministic_and_seeded() -> None:
    df = _split_frame(n_patients=30, per_patient=1)
    sids = df["SampleID"].tolist()
    coll = _coll()
    first = assign_splits(df, sids, coll, test_size=0.2, random_state=3)
    again = assign_splits(df, sids[::-1], coll, test_size=0.2, random_state=3)
    other = assign_splits(df, sids, coll, test_size=0.2, random_state=4)
    assert first == again
    assert first != other
    assert sum(v == "Ts" for v in first.values()) == 6


@pytest.mark.parametrize(
    ("n", "test_size", "expected_test"),
    [(1, 0.2, 0), (2, 0.2, 1), (5, 0.0, 0), (3, 0.99, 2)],
)
def test_random_split_edge_sizes(
    n: int, test_size: float, expected_test: int
) -> None:
    df = _split_frame(n_patients=n, per_patient=1)
    splits = assign_splits(
        df, df["SampleID"].tolist(), _coll(), test_size=test_size,
        random_state=0,
    )  # fmt: skip
    assert sum(v == "Ts" for v in splits.values()) == expected_test
    assert sum(v == "Tr" for v in splits.values()) == n - expected_test


def test_missing_group_value_splits_sample_alone() -> None:
    df = _split_frame(n_patients=4, per_patient=2)
    df.loc[0, "Patient"] = np.nan
    splits = assign_splits(
        df, df["SampleID"].tolist(), _coll(group_column="Patient"),
        test_size=0.5, random_state=0,
    )  # fmt: skip
    assert set(splits) == set(df["SampleID"])


# ---------------------------------------------------------------------------
# Case IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collection", "sid", "prefix", "expected"),
    [
        ("Abdomen Atlas 3", "123", True, "AbdomenAtlas3_123"),
        ("LLD-MMRI", "MR-391135", True, "LLD-MMRI_MR-391135"),
        ("MSWAL", "Tr_1", True, "MSWAL_Tr_1"),
        ("MSWAL", "a b/c.d", False, "abcd"),
    ],
)
def test_make_case_id(
    collection: str, sid: str, prefix: bool, expected: str
) -> None:
    assert make_case_id(collection, sid, prefix_with_collection=prefix) == expected


def test_make_case_id_rejects_empty() -> None:
    with pytest.raises(PlanError):
        make_case_id("C", "...", prefix_with_collection=True)


def test_case_id_collisions_fail(source: Source) -> None:
    for row in source.rows:
        if row["Collection"] == MSWAL:
            row["SampleID"] = row["SampleID"].replace("M", "A")
    source.save_csv()
    with pytest.raises(PlanError, match="shared by more than one sample"):
        _plan(source, prefix_with_collection=False)
    _plan(source, prefix_with_collection=True)  # prefix keeps them apart
