"""Integration tests for the imgnet download CLI."""

import csv
import pytest
from click.testing import CliRunner
from pathlib import Path

from imgnet.cli.__main__ import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_download_help(runner: CliRunner):
    """--help shows usage and mentions manifest/token."""
    result = runner.invoke(cli, ["download", "--help"])
    assert result.exit_code == 0
    assert "manifest" in result.output.lower() or "token" in result.output.lower()
    assert "output" in result.output.lower() or "-o" in result.output


def test_download_no_args_shows_help(runner: CliRunner):
    """No args in non-tty shows help and exit 0 (CLI design)."""
    result = runner.invoke(cli, ["download"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "output" in result.output.lower()


def test_download_from_manifest_file(runner: CliRunner, tmp_path: Path):
    """Download with manifest CSV path creates output dir (actual download may run)."""
    manifest_path = tmp_path / "manifest.csv"
    # Minimal manifest: Collection, SeriesInstanceUID (and any cols ImgNet.download expects)
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Collection", "SeriesInstanceUID", "PatientID", "StudyInstanceUID"])
        w.writerow(["4D-Lung", "1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985", "", ""])
    out_dir = tmp_path / "downloaded"
    result = runner.invoke(
        cli,
        ["download", str(manifest_path), "--output-dir", str(out_dir)],
    )
    assert result.exit_code == 0
    assert out_dir.exists()


def test_download_with_process_flag_accepts(runner: CliRunner, tmp_path: Path):
    """Download with --process flag is accepted (processing may run).

    Uses a manifest CSV (query tokens are not wired through download yet; a long
    token string is also mishandled as a filesystem path).
    """
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Collection", "SeriesInstanceUID", "PatientID", "StudyInstanceUID"])
        w.writerow(
            [
                "4D-Lung",
                "1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985",
                "",
                "",
            ]
        )
    out_dir = tmp_path / "processed"
    result = runner.invoke(
        cli,
        ["download", str(manifest_path), "--output-dir", str(out_dir), "--process"],
    )
    assert result.exit_code == 0
    assert (out_dir / "srcdata").exists()
    assert (out_dir / "procdata").exists()
