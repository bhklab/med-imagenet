import click
from imgtools.dicom import Interlacer
from pathlib import Path
import pandas as pd
import json
from rich.table import Table
from rich import print
from imgnet.collections_summary import Collections

import re
@click.command(no_args_is_help=False)
@click.help_option(
    "--help",
    "-h"
)
@click.option(
    "--body-part",
    "-b",
    type=str,
    multiple=True,
    default=None
)
@click.option(
    "--modality",
    "-m",
    type=str,
    multiple=True,
    default=None
)
def collections(body_part: list[str], modality: list[str]):
    """Display all supported collections."""
    collections = Collections()
    collections.display_collections(body_part=body_part, modality=modality)

    
            


