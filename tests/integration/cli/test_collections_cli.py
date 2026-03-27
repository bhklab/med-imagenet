"""Integration tests for the imgnet collections CLI."""

import json

import pytest
from click.testing import CliRunner

from imgnet.cli.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_collections_help(runner: CliRunner):
    """collections --help lists subcommands."""
    result = runner.invoke(cli, ["collections", "--help"])
    assert result.exit_code == 0
    assert "summary" in result.output


def test_collections_requires_subcommand(runner: CliRunner):
    """Bare collections prints help and exits non-zero."""
    result = runner.invoke(cli, ["collections"])
    assert result.exit_code != 0
    assert "summary" in result.output.lower() or "Usage" in result.output


def test_collections_summary(runner: CliRunner, store):
    """collections summary prints summary table."""
    result = runner.invoke(cli, ["collections", "summary"])
    assert result.exit_code == 0
    assert "Collection" in result.output or "collection" in result.output.lower()


def test_collections_summary_update(runner: CliRunner, store):
    """collections summary --update runs and exits 0."""
    result = runner.invoke(cli, ["collections", "summary", "--update"])
    assert result.exit_code == 0


def test_collections_info_default(runner: CliRunner, store):
    """collections info shows overview only by default."""
    name = store.collections[0]
    result = runner.invoke(cli, ["collections", "info", name])
    assert result.exit_code == 0
    assert "Collection:" in result.output
    assert "Modalities" in result.output
    assert "Supported query tags" not in result.output


def test_collections_info_tags(runner: CliRunner, store):
    """collections info --tags appends query tags table."""
    name = store.collections[0]
    result = runner.invoke(cli, ["collections", "info", name, "--tags"])
    assert result.exit_code == 0
    assert "Supported query tags" in result.output


def test_collections_info_json(runner: CliRunner, store):
    """collections info --json emits parseable JSON."""
    name = store.collections[0]
    result = runner.invoke(cli, ["collections", "info", name, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["collection"] == name
    assert "file_type" in data
    assert "modalities" in data
    assert "supported_query_tags" not in data


def test_collections_info_json_tags(runner: CliRunner, store):
    """collections info --json --tags includes supported_query_tags."""
    name = store.collections[0]
    result = runner.invoke(
        cli, ["collections", "info", name, "--json", "--tags"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "supported_query_tags" in data
    assert isinstance(data["supported_query_tags"], dict)


def test_collections_info_unknown(runner: CliRunner, store):
    """collections info fails for unknown collection with a hint."""
    result = runner.invoke(
        cli, ["collections", "info", "__not_a_collection__"]
    )
    assert result.exit_code != 0
    assert "Unknown collection" in result.output
    assert "IndexedDatasets.collections" in result.output

def test_collections_info_fuzzy_suggestion(runner: CliRunner, store):
    """Typo similar to a real name triggers Did you mean."""
    real = max(store.collections, key=len)
    if len(real) < 8:
        pytest.skip("need a long collection name for a reliable fuzzy match")
    typo = real[:-1]
    if typo in store.collections:
        typo = real[:-2]
    result = runner.invoke(cli, ["collections", "info", typo])
    assert result.exit_code != 0
    assert "Unknown collection" in result.output
    assert "Did you mean" in result.output
    assert real in result.output
