import click
from pathlib import Path

from imgnet.collections.store import IndexedDatasets


@click.command(no_args_is_help=False)
@click.help_option(
    "--help",
    "-h"
)
@click.option(
    "--update",
    "-u",
    is_flag=True,
    help="Update the collections summary."
)

def collections(update: bool):
    """Display all supported collections."""
    store = IndexedDatasets()
    store.display_summary(update)

    
            


