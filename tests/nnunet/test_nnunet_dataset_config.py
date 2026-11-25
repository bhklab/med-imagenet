"""Config validation for imgnet nnunet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from imgnet.nnunet.dataset_config import (
    DatasetConfigError,
    NnUNetDatasetConfig,
    Regex,
    load_dataset_config,
)
from tests.nnunet.helpers import Source, dataset_config_dict


def _validate(src: Source, **overrides: object) -> NnUNetDatasetConfig:
    return NnUNetDatasetConfig.model_validate(dataset_config_dict(src, **overrides))


def _collection(src: Source, index: int = 0, **changes: object) -> list:
    collections = dataset_config_dict(src)["collections"]
    collections[index].update(changes)
    return collections


def test_valid_config_and_folder_name(source: Source) -> None:
    config = _validate(source)
    assert config.dataset_folder_name == "Dataset007_Lesions"
    assert config.dataset_dir == source.root / "nnUNet_raw" / "Dataset007_Lesions"


@pytest.mark.parametrize(
    "labels",
    [
        {"tumor": 1},  # no background
        {"background": 1, "tumor": 0},  # background not 0
        {"background": 0, "tumor": 2},  # gap
        {"background": 0, "a": 1, "b": 1},  # duplicate value
    ],
)
def test_labels_must_be_consecutive_with_background(
    source: Source, labels: dict
) -> None:
    with pytest.raises(ValidationError):
        _validate(source, labels=labels, collections=[])


@pytest.mark.parametrize("name", ["Abdomen Lesions", "a/b", ""])
def test_dataset_name_rejects_unsafe_characters(
    source: Source, name: str
) -> None:
    with pytest.raises(ValidationError, match="dataset_name"):
        _validate(source, dataset_name=name)


@pytest.mark.parametrize("dataset_id", [0, 1000])
def test_dataset_id_range(source: Source, dataset_id: int) -> None:
    with pytest.raises(ValidationError):
        _validate(source, dataset_id=dataset_id)


def test_intensity_values_must_be_labels(source: Source) -> None:
    collections = _collection(source, intensity_map={"liver_lesion": 5})
    with pytest.raises(ValidationError, match=r"label values \[5\]"):
        _validate(source, collections=collections)


@pytest.mark.parametrize("key", ["liver", "1.5", "nan", ""])
def test_remap_keys_must_be_whole_numbers(source: Source, key: str) -> None:
    collections = _collection(source, 1, intensity_map={key: 1})
    with pytest.raises(ValidationError, match="whole-number mask values"):
        _validate(source, collections=collections)


def test_remap_keys_accept_integral_floats(source: Source) -> None:
    collections = _collection(source, 1, intensity_map={"1.0": 1, "2": 2})
    config = _validate(source, collections=collections)
    assert config.collections[1].intensity_map == {"1": 1, "2": 2}


def test_remap_keys_must_not_repeat_a_value(source: Source) -> None:
    collections = _collection(source, 1, intensity_map={"1": 1, "1.0": 2})
    with pytest.raises(ValidationError, match="mask value 1 more than once"):
        _validate(source, collections=collections)


@pytest.mark.parametrize(
    "field", [{"roi_column": "Label"}, {"allow_missing_rois": True}]
)
def test_combine_only_fields_rejected_on_remap(
    source: Source, field: dict
) -> None:
    collections = _collection(source, 1, **field)
    with pytest.raises(ValidationError, match="only apply to method 'combine'"):
        _validate(source, collections=collections)


def test_config_round_trips_through_model_dump(source: Source) -> None:
    config = _validate(source)
    again = NnUNetDatasetConfig.model_validate(config.model_dump())
    assert again == config


def test_combine_labels_must_be_positive(source: Source) -> None:
    collections = _collection(source, intensity_map={"liver_lesion": 0})
    with pytest.raises(ValidationError, match=">= 1"):
        _validate(source, collections=collections)


def test_unknown_keys_are_rejected(source: Source) -> None:
    collections = _collection(source, mask_selecter={"x": "y"})
    with pytest.raises(ValidationError, match="mask_selecter"):
        _validate(source, collections=collections)
    with pytest.raises(ValidationError, match="output_dir"):
        _validate(source, output_dir="somewhere")


def test_duplicate_collections_rejected(source: Source) -> None:
    first = dataset_config_dict(source)["collections"][0]
    with pytest.raises(ValidationError, match="more than once"):
        _validate(source, collections=[first, first])


def test_split_values_must_not_overlap(source: Source) -> None:
    split = {"column": "Split", "train": ["Tr"], "test": ["Tr"]}
    collections = _collection(source, 1, split=split)
    with pytest.raises(ValidationError, match="both train and test"):
        _validate(source, collections=collections)


def test_selector_value_forms(source: Source) -> None:
    selector = {"Phase": "C+V", "ImageID": [1, 2], "filepath": {"regex": "ct"}}
    collections = _collection(source, selector=selector)
    config = _validate(source, collections=collections)
    parsed = config.collections[0].selector
    assert parsed["Phase"] == "C+V"
    assert parsed["ImageID"] == [1, 2]
    assert parsed["filepath"] == Regex(regex="ct")


def test_invalid_regex_rejected(source: Source) -> None:
    collections = _collection(source, selector={"filepath": {"regex": "("}})
    with pytest.raises(ValidationError, match="invalid regex"):
        _validate(source, collections=collections)


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    sub = tmp_path / "configs"
    sub.mkdir()
    raw = dataset_config_dict(
        Source(tmp_path, tmp_path / "srcdata", tmp_path / "q.csv", []),
        input_dir="../srcdata",
        query_results_csv="q.csv",
        nnunet_raw_dir="raw",
    )
    path = sub / "config.json"
    path.write_text(json.dumps(raw))
    config = load_dataset_config(path)
    assert config.input_dir.resolve() == (tmp_path / "srcdata").resolve()
    assert config.query_results_csv == sub.resolve() / "q.csv"
    assert config.nnunet_raw_dir == sub.resolve() / "raw"


def test_random_state_must_be_non_negative(source: Source) -> None:
    with pytest.raises(ValidationError, match="random_state"):
        _validate(source, random_state=-1)


def test_load_config_reads_utf8(tmp_path: Path) -> None:
    raw = dataset_config_dict(
        Source(tmp_path, tmp_path / "srcdata", tmp_path / "q.csv", [])
    )
    raw["collections"][0]["collection"] = "Ünïcode Atlas"
    path = tmp_path / "config.json"
    path.write_bytes(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    config = load_dataset_config(path)
    assert config.collections[0].collection == "Ünïcode Atlas"


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetConfigError, match="Could not read"):
        load_dataset_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(DatasetConfigError, match="Could not read"):
        load_dataset_config(bad)
    bad.write_text("{}")
    with pytest.raises(DatasetConfigError, match="Invalid config"):
        load_dataset_config(bad)
