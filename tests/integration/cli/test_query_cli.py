"""Integration tests for the imgnet query CLI."""

import json
import pandas as pd
import pytest
from click.testing import CliRunner
from pathlib import Path

from imgnet.cli.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_query_cli_from_json(runner: CliRunner, tmp_path: Path):
    """Query from JSON input path runs and writes results."""
    result = runner.invoke(
        cli,
        [
            "query",
            str(tmp_path),
            "-i",
            "tests/test_dir/valid_query.json",
        ],
    )
    assert result.exit_code == 0, f"Loading from json failed: {result.output}"
    assert (tmp_path / "query_results.csv").exists()
    assert (tmp_path / "valid_query.json").exists()
    assert (tmp_path / "valid_query_schema.json").exists()


def test_query_cli_options(runner: CliRunner, tmp_path: Path):
    """Query from CLI options (-c -m -r) runs and writes results."""
    result = runner.invoke(
        cli,
        [
            "query",
            str(tmp_path),
            "-c",
            "4D-Lung",
            "-m",
            "CT,RTSTRUCT",
            "-r",
            '{"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069", "RTSTRUCT": "Modality == CT"}',
        ],
    )
    assert result.exit_code == 0, f"Query from cli failed: {result.output}"
    assert (tmp_path / "query_results.csv").exists()


@pytest.mark.parametrize(
    "collections, modalities, rules, expected_series",
    [
        (
            ["4D-Lung"],
            "CT",
            {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"},
            ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"],
        ),
        (
            ["4D-Lung", "Adrenal-ACC-Ki67-Seg"],
            "CT",
            {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"},
            ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"],
        ),
        (
            ["all"],
            "CT",
            {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"},
            ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"],
        ),
        (
            ["all"],
            "CT,RTSTRUCT",
            {
                "CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069",
                "RTSTRUCT": "Modality == CT",
            },
            ["1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069"],
        ),
    ],
)
def test_query_cli_process(
    runner: CliRunner,
    tmp_path: Path,
    collections,
    modalities,
    rules,
    expected_series,
):
    """Query via CLI returns expected SeriesInstanceUIDs (same scenarios as former test_query_process)."""
    args = ["query", str(tmp_path), "-m", modalities, "-r", json.dumps(rules)]
    if isinstance(collections, list):
        for c in collections:
            args.extend(["-c", c])
    else:
        args.extend(["-c", collections])

    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    df = pd.read_csv(tmp_path / "query_results.csv")
    assert df["SeriesInstanceUID"].tolist() == expected_series


def test_query_cli_invalid_rule_exits_nonzero(runner: CliRunner, tmp_path: Path):
    """Invalid rule string causes non-zero exit."""
    result = runner.invoke(
        cli,
        [
            "query",
            str(tmp_path),
            "-c",
            "4D-Lung",
            "-m",
            "CT",
            "-r",
            '{"CT": "Invalid Rule"}',
        ],
    )
    assert result.exit_code != 0


def test_query_cli_unknown_collection_exits_nonzero(runner: CliRunner, tmp_path: Path):
    """Unknown collection name causes non-zero exit."""
    result = runner.invoke(
        cli,
        [
            "query",
            str(tmp_path),
            "-c",
            "Unsupported_Collection",
            "-m",
            "all",
        ],
    )
    assert result.exit_code != 0
