"""Case selection: selectors, ambiguity, skips, splits and case IDs (no image I/O)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from imgnet.nnunet.dataset_config import NnUNetDatasetConfig, Regex
from imgnet.nnunet.case_selection import (
    CaseSelectionError,
    assign_splits,
    make_case_id,
    select_cases,
    read_query_results,
    select_rows,
)
from tests.nnunet.helpers import (
    ATLAS,
    MSWAL,
    Source,
    dataset_config_dict,
    liver_mask,
    make_dataset_config,
    write_volume,
)


def _select(src: Source, **overrides: object):  # noqa: ANN202
    config = make_dataset_config(src, **overrides)
    return select_cases(read_query_results(src.csv), config)


def _collections(src: Source, *changes: dict) -> list[dict]:
    collections = dataset_config_dict(src)["collections"]
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
            "ImageID": ["1", "2.0", "3", "10"],
            "Score": ["0.5", "1.25", None, "x"],
            "Flag": ["True", "False", "1", None],
        }
    )
    assert select_rows(df, {}).tolist() == [True] * 4
    assert select_rows(df, {"Phase": "C+V"}).tolist() == [1, 0, 0, 0]
    assert select_rows(df, {"Phase": ["C+V", "DWI"]}).tolist() == [1, 0, 0, 1]
    # Whole numbers match "2" and "2.0" (imgnet query writes int columns as
    # floats), but 1 must not match "10".
    assert select_rows(df, {"ImageID": 1}).tolist() == [1, 0, 0, 0]
    assert select_rows(df, {"ImageID": 2}).tolist() == [0, 1, 0, 0]
    assert select_rows(df, {"ImageID": [2, 10]}).tolist() == [0, 1, 0, 1]
    # A string is compared as text: "2" doesn't match "2.0".
    assert select_rows(df, {"ImageID": "2"}).tolist() == [0, 0, 0, 0]
    # Floats compare numerically; text that isn't a number never matches.
    assert select_rows(df, {"Score": 1.25}).tolist() == [0, 1, 0, 0]
    assert select_rows(df, {"Score": 0.50}).tolist() == [1, 0, 0, 0]
    # Booleans are text, and True must not match "1".
    assert select_rows(df, {"Flag": True}).tolist() == [1, 0, 0, 0]
    # Regex is searched; missing values never match.
    regex = {"Phase": Regex(regex=r"^C\+")}
    assert select_rows(df, regex).tolist() == [1, 1, 0, 0]
    both = {"Phase": Regex(regex="C"), "ImageID": ["2.0", "10"]}
    assert select_rows(df, both).tolist() == [0, 1, 0, 0]


# ---------------------------------------------------------------------------
# Resolving cases
# ---------------------------------------------------------------------------


def test_plan_resolves_cases_and_skips(source: Source) -> None:
    selection = _select(source)
    by_id = {c.case_id: c for c in selection.cases}
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

    assert [(s.sample_id, s.reason) for s in selection.skipped] == [
        ("A4", "missing ROIs ['kidney_lesion']")
    ]


def test_allow_missing_rois_keeps_partial_samples(source: Source) -> None:
    collections = _collections(source, {"allow_missing_rois": True})
    selection = _select(source, collections=collections)
    a4 = next(c for c in selection.cases if c.sample_id == "A4")
    assert a4.labels == [1]
    assert not selection.skipped


def test_ambiguous_scans_fail_until_selected(source: Source) -> None:
    # Give M1 a second scan in another phase, like LLD-MMRI.
    for row in source.rows:
        if row["Collection"] == MSWAL:
            row["Phase"] = "C+V"
    extra = "imagesTr/M1_arterial.nii.gz"
    write_volume(source.srcdata / MSWAL / extra, liver_mask())
    source.add(MSWAL, "M1", "scan", extra, Split="Tr", Phase="C+A")
    source.save_csv()

    with pytest.raises(CaseSelectionError, match=r"MSWAL: 1 samples match more than one scan"):
        _select(source)

    collections = _collections(source, {}, {"selector": {"Phase": "C+V"}})
    selection = _select(source, collections=collections)
    m1 = next(c for c in selection.cases if c.sample_id == "M1")
    assert m1.scan.name == "M1_0000.nii.gz"


def test_ambiguous_masks_fail_until_selected(source: Source) -> None:
    extra = "labelsTr/M1_anatomy.nii.gz"
    write_volume(source.srcdata / MSWAL / extra, liver_mask())
    source.add(MSWAL, "M1", "mask", extra, Split="Tr", mask_type="anatomy")
    source.save_csv()

    with pytest.raises(CaseSelectionError, match="more than one mask"):
        _select(source)

    selector = {"mask_selector": {"filepath": {"regex": r"M\d\.nii"}}}
    collections = _collections(source, {}, selector)
    selection = _select(source, collections=collections)
    m1 = next(c for c in selection.cases if c.sample_id == "M1")
    assert [p.name for p in m1.masks] == ["M1.nii.gz"]


def test_duplicate_roi_masks_are_ambiguous(source: Source) -> None:
    source.add(ATLAS, "A1", "mask", "masks/A1/aorta.nii.gz",
               ROINames="liver_lesion", PatientID="P1")  # fmt: skip
    source.save_csv()
    with pytest.raises(CaseSelectionError, match=r"A1 \(liver_lesion\)"):
        _select(source)


def test_missing_files_are_skipped(source: Source) -> None:
    (source.srcdata / MSWAL / "labelsTr/M2.nii.gz").unlink()
    selection = _select(source)
    assert "M2" not in {c.sample_id for c in selection.cases}
    reasons = {s.sample_id: s.reason for s in selection.skipped}
    assert reasons["M2"].startswith("file not found:")


def test_samples_without_scan_or_mask_are_skipped(source: Source) -> None:
    source.add(MSWAL, "M9", "scan", "imagesTr/M9.nii.gz", Split="Ts")
    source.add(MSWAL, "M8", "mask", "labelsTr/M8.nii.gz", Split="Ts")
    source.save_csv()
    reasons = {s.sample_id: s.reason for s in _select(source).skipped}
    assert reasons["M9"] == "no mask matched"
    assert reasons["M8"] == "no scan matched"


def test_missing_columns_and_collections(source: Source) -> None:
    collections = _collections(source, {"scan_selector": {"Phase": "x"}})
    with pytest.raises(CaseSelectionError, match=r"columns \['Phase'\] are not in"):
        _select(source, collections=collections)

    collections = _collections(source, {"collection": "Nope"})
    with pytest.raises(CaseSelectionError, match=r"\['Nope'\] are configured"):
        _select(source, collections=collections)

    df = read_query_results(source.csv).drop(columns=["file_type"])
    with pytest.raises(CaseSelectionError, match="missing required columns"):
        select_cases(df, make_dataset_config(source))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_column_is_used(source: Source) -> None:
    splits = {c.sample_id: c.split for c in _select(source).cases}
    assert (splits["M1"], splits["M2"], splits["M3"]) == ("Tr", "Tr", "Ts")


def test_split_column_unmapped_value_fails(source: Source) -> None:
    for row in source.rows:
        if row.get("Split") == "Ts":
            row["Split"] = "Va"
    source.save_csv()
    with pytest.raises(CaseSelectionError, match=r"values \['Va'\] are not listed"):
        _select(source)


def test_split_column_conflict_fails(source: Source) -> None:
    for row in source.rows:
        if row["SampleID"] == "M1" and row["file_type"] == "mask":
            row["Split"] = "Ts"
    source.save_csv()
    with pytest.raises(CaseSelectionError, match="more than one 'Split' value"):
        _select(source)


def test_missing_split_column_falls_back_to_random(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    split = {"column": "OfficialSplit", "train": ["Tr"]}
    collections = _collections(source, {"split": split})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        selection = _select(source, collections=collections)
    atlas = [c for c in selection.cases if c.collection == ATLAS]
    assert {c.split for c in atlas} == {"Tr", "Ts"}
    assert "'OfficialSplit' is not in the query results" in caplog.text
    assert "using a random split" in caplog.text


def test_split_column_typo_suggests_close_names(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    collections = _collections(
        source, {}, {"split": {"column": "Splt", "train": ["Tr"]}}
    )
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        _select(source, collections=collections)
    assert "did you mean ['Split']" in caplog.text


def test_split_column_empty_for_collection_is_reported_as_such(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    # 'Split' exists (MSWAL rows) but has no values for Abdomen Atlas 3.
    split = {"column": "Split", "train": ["Tr"]}
    collections = _collections(source, {"split": split})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        _select(source, collections=collections)
    assert "'Split' has no values for this collection" in caplog.text


def test_split_column_warns_that_group_column_is_ignored(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    collections = _collections(source, {}, {"group_column": "SampleID"})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        selection = _select(source, collections=collections)
    splits = {c.sample_id: c.split for c in selection.cases}
    assert splits["M3"] == "Ts"  # still from the Split column
    assert "group_column 'SampleID' only applies to random splits" in caplog.text


def test_split_fallback_uses_group_column(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    split = {"column": "OfficialSplit", "train": ["Tr"]}
    collections = _collections(source, {"split": split})
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        selection = _select(source, collections=collections)
    atlas = {c.sample_id: c.split for c in selection.cases if c.collection == ATLAS}
    assert atlas["A1"] == atlas["A2"]  # same patient P1
    assert "grouped by 'PatientID'" in caplog.text


def _split_frame(n_patients: int, per_patient: int) -> pd.DataFrame:
    rows = [
        {"SampleID": f"S{p}_{i}", "Patient": f"P{p}"}
        for p in range(n_patients)
        for i in range(per_patient)
    ]
    return pd.DataFrame(rows)


def _coll(collection: str = "C", **changes: object):  # noqa: ANN202
    config = NnUNetDatasetConfig.model_validate(
        {
            "input_dir": ".",
            "query_results_csv": "q.csv",
            "nnunet_raw_dir": ".",
            "dataset_id": 1,
            "dataset_name": "X",
            "labels": {"background": 0, "a": 1},
            "collections": [
                {
                    "collection": collection,
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


def test_missing_group_value_splits_sample_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 10 samples, none with a patient value. If they were lumped into one
    # group, everything would go to training; each must be its own group.
    df = _split_frame(n_patients=10, per_patient=1)
    df["Patient"] = np.nan
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        splits = assign_splits(
            df, df["SampleID"].tolist(), _coll(group_column="Patient"),
            test_size=0.5, random_state=0,
        )  # fmt: skip
    assert sum(v == "Ts" for v in splits.values()) == 5
    assert "10 samples have no 'Patient' value" in caplog.text


def test_random_split_is_stable_when_samples_change() -> None:
    df = _split_frame(n_patients=100, per_patient=1)
    sids = df["SampleID"].tolist()
    coll = _coll()
    before = assign_splits(df, sids, coll, test_size=0.2, random_state=7)
    for removed in sids[:20]:
        kept = [s for s in sids if s != removed]
        after = assign_splits(df, kept, coll, test_size=0.2, random_state=7)
        moved = [s for s in kept if after[s] != before[s]]
        assert len(moved) <= 1, (removed, moved)
        # Removing a sample never pushes a test sample into training.
        assert not any(before[s] == "Ts" and after[s] == "Tr" for s in moved)


def test_random_split_seeds_and_collections_are_independent() -> None:
    df = _split_frame(n_patients=200, per_patient=1)
    sids = df["SampleID"].tolist()

    def test_set(seed: int, collection: str = "C") -> set[str]:
        splits = assign_splits(
            df, sids, _coll(collection=collection), test_size=0.2,
            random_state=seed,
        )  # fmt: skip
        return {s for s, v in splits.items() if v == "Ts"}

    base = test_set(1)
    assert len(base) == 40
    # Independent 20% test sets overlap by about 8 samples; a weak hash
    # (e.g. CRC) gives near-identical or disjoint sets instead.
    for other in (test_set(2), test_set(3), test_set(1, collection="D")):
        assert len(other) == 40
        assert 0 < len(base & other) < 20


def test_split_size_is_not_inflated_by_float_error() -> None:
    df = _split_frame(n_patients=100, per_patient=1)
    splits = assign_splits(
        df, df["SampleID"].tolist(), _coll(), test_size=0.07, random_state=0
    )
    assert sum(v == "Ts" for v in splits.values()) == 7  # not ceil(7.0000001)


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
    with pytest.raises(CaseSelectionError):
        make_case_id("C", "...", prefix_with_collection=True)


def test_case_id_collisions_fail(source: Source) -> None:
    for row in source.rows:
        if row["Collection"] == MSWAL:
            row["SampleID"] = row["SampleID"].replace("M", "A")
    source.save_csv()
    with pytest.raises(CaseSelectionError, match="shared by more than one sample"):
        _select(source, prefix_with_collection=False)
    _select(source, prefix_with_collection=True)  # prefix keeps them apart


def test_case_ids_differing_only_by_letter_case_fail(source: Source) -> None:
    # MSWAL_M1 and MSWAL_m1 would be the same file on macOS and Windows.
    for row in source.rows:
        if row["Collection"] == MSWAL and row["SampleID"] == "M2":
            row["SampleID"] = "m1"
    source.save_csv()
    with pytest.raises(CaseSelectionError, match="ignoring letter case"):
        _select(source)


# ---------------------------------------------------------------------------
# Configuration mistakes that must not pass silently
# ---------------------------------------------------------------------------


def test_misspelled_roi_fails_with_suggestions(source: Source) -> None:
    for allow_missing in (False, True):
        collections = _collections(
            source,
            {
                "intensity_map": {"liver_lesoin": 1, "kidney_lesion": 2},
                "allow_missing_rois": allow_missing,
            },
        )
        with pytest.raises(CaseSelectionError) as info:
            _select(source, collections=collections)
        message = str(info.value)
        assert "ROIs ['liver_lesoin'] have no masks" in message
        assert "'liver_lesion'" in message  # close match
        assert "'aorta'" in message  # available ROI names are listed


def test_collection_without_cases_fails(source: Source) -> None:
    collections = _collections(source, {}, {"mask_selector": {"Split": "x"}})
    with pytest.raises(CaseSelectionError, match="MSWAL: no cases selected"):
        _select(source, collections=collections)


def test_wrong_input_dir_gets_a_hint(source: Source) -> None:
    with pytest.raises(CaseSelectionError) as info:
        _select(source, input_dir=str(source.root))  # parent of srcdata
    message = str(info.value)
    assert "no cases selected" in message
    assert "check that input_dir points at the download folder" in message
    assert "'Abdomen Atlas 3'" in message


def test_no_training_cases_fails(source: Source) -> None:
    split = {"column": "Split", "train": ["none"], "test": ["Tr", "Ts"]}
    collections = [_collections(source, {}, {"split": split})[1]]
    with pytest.raises(CaseSelectionError, match="No training cases"):
        _select(source, collections=collections)


def test_modality_warnings(
    source: Source, caplog: pytest.LogCaptureFixture
) -> None:
    for row in source.rows:
        row["Modality"] = "CT" if row["Collection"] == ATLAS else "MR"
    source.save_csv()
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        _select(source)
    assert "mixed modalities {'CT': 3, 'MR': 3}" in caplog.text

    caplog.clear()
    collections = [_collections(source)[1]]
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        _select(source, collections=collections)
    assert "Selected scans are MR but channel_name is 'CT'" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="imgnet"):
        _select(source, collections=collections, channel_name="mr")
    assert "channel_name" not in caplog.text


@pytest.mark.parametrize(
    "content", [b"", b"\xff\xfe\x00bad", b'a,b\n"unterminated']
)
def test_unreadable_query_results_fail_cleanly(
    tmp_path, content: bytes  # noqa: ANN001
) -> None:
    path = tmp_path / "q.csv"
    path.write_bytes(content)
    with pytest.raises(CaseSelectionError, match="Could not read query results"):
        read_query_results(path)
