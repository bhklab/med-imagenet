import click
from pathlib import Path

from imgnet.collections.store import IndexedDatasets


@click.command(no_args_is_help=False)
@click.help_option(
    "--help",
    "-h"
)

def collections():
    """Display all supported collections."""
    store = IndexedDatasets(Path.cwd() / "indexed_datasets")
    store.display_summary()

    
            


