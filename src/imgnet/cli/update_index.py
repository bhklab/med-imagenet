from pathlib import Path

import click

from imgnet.collections.store import IndexedDatasets


@click.command(no_args_is_help=False)
@click.option(
    "--path",
    "-p",
    "path_option",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Path to the indexed_datasets directory. Defaults to $IMGNET_INDEX_DIR/indexed_datasets or platform user data dir.",
)

def update_index(
    path_option: Path | None,
) -> None:
    """Update the indexed datasets: download from Hugging Face.

    Examples:

      imgnet update-index --path /custom/indexed_datasets
    """
    store = IndexedDatasets(path=path_option, force_download=True)