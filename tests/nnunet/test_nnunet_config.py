"""Config validation for imgnet nnunet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from imgnet.nnunet.config import (
    ConfigError,
    NnUNetConfig,
    Regex,
    load_config,
)
from tests.nnunet.helpers import Source, config_dict


def _validate(src: Source, **overrides: object) -> NnUNetConfig:
    return NnUNetConfig.model_validate(config_dict(src, **overrides))


def _collection(src: Source, index: int = 0, **changes: object) -> list:
    collections = config_dict(src)["collections"]
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


def test_remap_keys_must_be_integers(source: Source) -> None:
    collections = _collection(source, 1, intensity_map={"liver": 1})
    with pytest.raises(ValidationError, match="integer mask values"):
        _validate(source, collections=collections)


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
    first = config_dict(source)["collections"][0]
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
    raw = config_dict(
        Source(tmp_path, tmp_path / "srcdata", tmp_path / "q.csv", []),
        input_dir="../srcdata",
        query_results_csv="q.csv",
        nnunet_raw_dir="raw",
    )
    path = sub / "config.json"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config.input_dir.resolve() == (tmp_path / "srcdata").resolve()
    assert config.query_results_csv == sub.resolve() / "q.csv"
    assert config.nnunet_raw_dir == sub.resolve() / "raw"


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="Could not read"):
        load_config(bad)
    bad.write_text("{}")
    with pytest.raises(ConfigError, match="Invalid config"):
        load_config(bad)
