"""Integration tests for the imgnet collections CLI."""

import pytest
from click.testing import CliRunner

from imgnet.cli.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_collections_help(runner: CliRunner):
    """--help shows usage and update option."""
    result = runner.invoke(cli, ["collections", "--help"])
    assert result.exit_code == 0
    assert "update" in result.output.lower() or "-u" in result.output


def test_collections_list(runner: CliRunner, store):
    """collections (no --update) runs and prints (uses existing summary or builds)."""
    result = runner.invoke(cli, ["collections"])
    assert result.exit_code == 0
    # Should show table-like output (Collection, Modalities, etc.)
    assert "Collection" in result.output or "collection" in result.output.lower()


def test_collections_update(runner: CliRunner, store):
    """collections --update runs and exits 0."""
    result = runner.invoke(cli, ["collections", "--update"])
    assert result.exit_code == 0
