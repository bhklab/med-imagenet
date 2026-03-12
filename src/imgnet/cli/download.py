import click
from pathlib import Path
import pandas as pd
import sys

from imgnet.imgnet import ImgNet
from imgnet.collections.store import IndexedDatasets
from imgnet.loggers import logger

@click.command()
@click.option(
    "--manifest_path",
    "-m",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    default=None,
    help="Path to the manifest file."
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
@click.pass_context
def download(
    ctx: click.Context,
    manifest_path: Path | None,
    output_path: Path | None,
) -> None:
    """Download the instances from the manifest.
    """

    if manifest_path is None and not sys.stdin.isatty():
        manifest_path = sys.stdin.read().strip()
        logger.info("Reading manifest path from stdin: %s", manifest_path)

    if manifest_path is None:
        click.echo(ctx.get_help())
        ctx.exit(0)
        return
    manifest_path = Path(manifest_path)

    if output_path is not None:
        output_path = Path(output_path)
    if output_path is None:
        logger.warning("No output path provided, using default output path: %s", manifest_path.parent / "downloaded_data")
        output_path = manifest_path.parent / "downloaded_data"
    output_path.mkdir(exist_ok=True, parents=True)

    with open(manifest_path, "r") as f:
        manifest = pd.read_csv(f)
    
    store = IndexedDatasets()
    imgnet = ImgNet(output_path, store=store)
    imgnet.download(manifest)
