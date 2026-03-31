import click
from pathlib import Path
import pandas as pd
import sys
from datetime import datetime

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
    "--output-dir",
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
@click.option(
    "--delete-srcdata",
    is_flag=True,
    help="Delete the source data after processing."
)

@click.pass_context
def download(
    ctx: click.Context,
    manifest_or_token: str | None,
    output_dir: Path | None,
    process: bool,
    delete_srcdata: bool,
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
        with path.open("r") as f:
            manifest = pd.read_csv(f)
        logger.info(f"Downloading from manifest file: {path}")
    else:
        valid_query = ValidQuery.from_token(manifest_or_token)
        manifest = valid_query.process(store)
        logger.info(f"Downloading from query token: {manifest_or_token}")

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path.cwd() / f"imgnet_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.warning("No output path provided, using: %s", output_dir)

    srcdata_path = output_dir / "srcdata"
    srcdata_path.mkdir(exist_ok=True, parents=True)

    imgnet = ImgNet(srcdata_path, store=store)
    imgnet.download(manifest)

    if process:
        from imgtools.autopipeline import Autopipeline
        from imgnet.collections.source import FileType
        import shutil
        procdata_path = output_dir / "procdata"
        procdata_path.mkdir(exist_ok=True, parents=True)

        for collection in manifest["Collection"].unique():
            if store.file_type(collection) == FileType.DICOM:
                logger.info(f"Processing DICOM files in collection using imgtools: {collection}")
                pipeline = Autopipeline(
                    input_directory=srcdata_path / collection,
                    output_directory=procdata_path / collection,
                )
                pipeline.run()

                if delete_srcdata:
                    logger.warning(f"Flag --delete-srcdata set. Deleting source data for collection: {collection}")
                    if (srcdata_path / collection).exists():
                        shutil.rmtree(srcdata_path / collection)
                    if (srcdata_path / ".imgtools").exists():
                        shutil.rmtree(srcdata_path / ".imgtools")
            