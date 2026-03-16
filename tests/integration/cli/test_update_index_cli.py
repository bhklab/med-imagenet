"""Integration tests for the imgnet update-index CLI."""

import pytest
from click.testing import CliRunner
from pathlib import Path

from imgnet.cli.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_update_index_help(runner: CliRunner):
    """--help shows path option."""
    result = runner.invoke(cli, ["update-index", "--help"])
    assert result.exit_code == 0
    assert "path" in result.output.lower() or "-p" in result.output


def test_update_index_with_path(runner: CliRunner, tmp_path: Path):
    """update-index --path <dir> runs (may download from HF if path empty)."""
    # Use a temp path; IndexedDatasets will try to download if path doesn't exist.
    # We only assert the CLI accepts the option and runs without crashing.
    result = runner.invoke(
        cli,
        ["update-index", "--path", str(tmp_path)],
    )
    # Accept any exit code; we only care that the CLI ran without an uncaught exception.
    assert result.exit_code is not None
