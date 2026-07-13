from pathlib import Path

import click

from imgnet.collections.store import IndexedDatasets


@click.command(
    no_args_is_help=False,
    epilog="""
    Examples:

    Update the default ImgNet index:

        imgnet update-index

    Update an index stored at a custom location:

        imgnet update-index --path /custom/indexed_datasets
    """
)
@click.option(
    "--path",
    "-p",
    "path_option",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help=(
        "Path where the ImgNet indexed datasets will be stored. "
        "Defaults to $IMGNET_INDEX_DIR/indexed_datasets or the platform "
        "user data directory if the environment variable is not set."
    ),
)
def update_index(
    path_option: Path | None,
) -> None:
    """Download and update the local ImgNet dataset index.

    The index contains metadata describing available imaging collections,
    including information required for searching and downloading datasets.

    By default, the index is stored in the ImgNet user data directory.
    Use --path to store or update the index in a custom location.
    """
    store = IndexedDatasets(path=path_option, force_download=True)