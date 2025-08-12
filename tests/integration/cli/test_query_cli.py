import pytest
from click.testing import CliRunner
from pathlib import Path
import shutil

from imgnet.cli.query import query as query_cli


def test_query_cli():
    runner = CliRunner()
    result = runner.invoke(
        query_cli,
        [
            "tests/test_dir",
            "-i"
            "tests/test_dir/valid_query.json"
        ]
    )

    assert result.exit_code==0, f"Loading from json failed: {result.output}"

    result = runner.invoke(
        query_cli,
        [
            "tests/test_dir",
            "-c",
            "4D-Lung",
            "-m",
            "CT,RTSTRUCT",
            "-r",
            '{"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069", "RTSTRUCT": "Modality == CT"}'
        ]
    )
    assert result.exit_code==0, f"Query from cli failed: {result.output}"

    result = runner.invoke(
        query_cli,
        [
            "tests/test_dir",
            "-c",
            "4D-Lung",
            "-m",
            "CT,RTSTRUCT",
            "-r",
            '{"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069", "RTSTRUCT": "Modality == CT"}'
            "--download"
        ]
    )
    shutil.rmtree("tests/test_dir/raw_data", ignore_errors=True)
    assert result.exit_code==0, f"Query with --download failed: {result.output}"

    



