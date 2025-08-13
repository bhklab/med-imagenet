import click

@click.command(no_args_is_help=False)
@click.help_option(
    "--help",
    "-h"
)
def updatedb():
    """Update the SUPPORTED_COLLECTIONS internal variable."""
    from glob import glob
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent.parent.parent.parent
    print(f"package location: {package_dir}")
    supported_collections = [f"\"{Path(folder).name}\""for folder in glob(f"{package_dir}/indexed_datasets/.imgtools/*")]
    with open(package_dir / "src" / "imgnet" / "supported_collections.py", "w") as f:
        f.write(f"SUPPORTED_COLLECTIONS = [\n{',\n'.join(supported_collections)}\n]")
    print("SUPPORTED_COLLECTIONS successfully updated.")
    

