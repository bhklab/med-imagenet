import click
from imgnet import __version__
from .query import query
from .collections import collections

@click.group(no_args_is_help=True)
@click.version_option(
    version=__version__,
    package_name="imgnet",
    prog_name="imgnet",
    message="%(package)s:%(prog)s:%(version)s",
)
@click.help_option("-h", "--help")
def cli() -> None:
    """IMGNET CLI - toolkit for facilitating access to standardized medical imaging datasets for cancer research and clinical AI applications."""
    pass

cli.add_command(query)
cli.add_command(collections)

if __name__ == "__main__":
    cli()