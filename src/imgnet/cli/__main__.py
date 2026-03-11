import click

from imgnet import __version__

from . import set_log_verbosity
from .query import query
from .collections import collections
from .update_index import update_index
from .sectioned_group import SectionedGroup, CommandRegistry

# Create a shared registry
registry = CommandRegistry()

# Register groups and commands
registry.create_group("core commands", "Main subcommands for the imgnet package.")
registry.add('core commands', query)

registry.create_group("utilities", "Tools for working with imgnet collections.")
registry.add("utilities", collections)
registry.add("utilities", update_index)

@click.group(cls=SectionedGroup, registry=registry, no_args_is_help=True)
@set_log_verbosity()
@click.version_option(
    version=__version__,
    package_name="imgnet",
    prog_name="imgnet",
    message="%(package)s:%(prog)s:%(version)s",
)
@click.help_option("-h", "--help")
def cli(verbose: int, quiet: bool) -> None:
    """IMGNET CLI - toolkit for facilitating access to standardized medical imaging datasets for cancer research and clinical AI applications."""
    pass

cli.add_registry(registry)

if __name__ == "__main__":
    cli()