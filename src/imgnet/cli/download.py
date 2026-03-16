import click
from pathlib import Path
import pandas as pd
import sys

from imgnet.imgnet import ImgNet
from imgnet.collections.store import IndexedDatasets
from imgnet.loggers import logger
from imgnet.query import ValidQuery

@click.command()
@click.argument(
    "manifest_or_token",
    type=click.STRING,
    required=False,
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    default=None,
    help="Path to the output directory."
)
@click.option(
    "--process",
    "-p",
    is_flag=True,
    help="Process all dicom files in the output directory using imgtools."
)

@click.pass_context
def download(
    ctx: click.Context,
    manifest_or_token: str | None,
    output_path: Path | None,
    process: bool,
) -> None:
    """Download instances from a manifest file or a query token.

    MANIFEST_OR_TOKEN: Path to a manifest CSV, or a query token from `imgnet query`.
    """
    if manifest_or_token is None and not sys.stdin.isatty():
        manifest_or_token = sys.stdin.read().strip()

    if not manifest_or_token:
        click.echo(ctx.get_help())
        ctx.exit(0)
        return

    store = IndexedDatasets()
    path = Path(manifest_or_token)
    if path.is_file():
        with open(path, "r") as f:
            manifest = pd.read_csv(f)
        default_output = path.parent / "downloaded_data"
        logger.info(f"Downloading from manifest file: {path}")
    else:
        valid_query = ValidQuery.from_token(manifest_or_token)
        manifest = valid_query.process(store)
        default_output = Path.cwd() / "downloaded_data"
        logger.info(f"Downloading from query token: {manifest_or_token}")

    if output_path is not None:
        output_path = Path(output_path)
    else:
        logger.warning("No output path provided, using: %s", default_output)
        output_path = default_output

    if process:
        output_path = output_path / "raw"
    output_path.mkdir(exist_ok=True, parents=True)

    imgnet = ImgNet(output_path, store=store)
    imgnet.download(manifest)

    if process:
        logger.info("Processing all dicom files in the output directory using imgtools.")
        from imgtools.autopipeline import Autopipeline

        pipeline = Autopipeline(
            input_directory=output_path,
            output_directory=output_path.parent / "processed_dicoms",
        )
        pipeline.run()
