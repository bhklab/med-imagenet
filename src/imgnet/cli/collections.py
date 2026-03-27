import json
from textwrap import fill

import click
from rich import print as rprint
from rich.table import Table

from imgnet.collections.store import IndexedDatasets


def _build_collection_info_payload(
    store: IndexedDatasets, collection: str, *, include_tags: bool
) -> dict:
    col = store.get_collection(collection)
    summ = col.summary
    modalities = summ.get("Modalities", [])
    if not isinstance(modalities, list):
        modalities = [str(modalities)]
    body_parts = summ.get("BodyPartsExamined", [])
    if not isinstance(body_parts, list):
        body_parts = [str(body_parts)]
    payload: dict = {
        "collection": collection,
        "file_type": str(summ["File Type"]),
        "source": str(summ["Source"]),
        "rows": len(col.index),
        "size_gb": float(summ["Size"]),
        "modalities": modalities,
        "body_parts_examined": body_parts,
        "description": col.description.strip(),
    }
    if include_tags:
        payload["supported_query_tags"] = store.supported_query_tags(collection)
    return payload


def _print_collection_overview(store: IndexedDatasets, collection: str) -> None:
    data = _build_collection_info_payload(
        store, collection, include_tags=False
    )
    table = Table(title=f"Collection: {data['collection']}")
    table.add_column("Field", justify="left", style="cyan")
    table.add_column("Value", justify="left")
    table.add_row("File type", data["file_type"])
    table.add_row("Source", data["source"])
    table.add_row("Rows", str(data["rows"]))
    table.add_row("Size (est.)", f"{data['size_gb']:.2f} GB")
    table.add_row(
        "Modalities",
        ", ".join(data["modalities"]) if data["modalities"] else "—",
    )
    table.add_row(
        "Body parts examined",
        ", ".join(data["body_parts_examined"])
        if data["body_parts_examined"]
        else "—",
    )
    desc = data["description"]
    table.add_row(
        "Description",
        fill(desc, width=88) if desc else "—",
    )
    rprint(table)


def _print_supported_query_tags(store: IndexedDatasets, collection: str) -> None:
    supported_tags = store.supported_query_tags(collection)
    table = Table(title=f"Supported query tags - {collection}")
    table.add_column("Modality", justify="left")
    table.add_column("Tags", justify="left")
    first = True
    for modality, tags in supported_tags.items():
        if not first:
            table.add_row("", "")
        table.add_row(modality, ", ".join(tags))
        first = False
    rprint(table)


def _print_collections_summary(store: IndexedDatasets, update: bool) -> None:
    table = Table(title="Collections Summary")
    table.add_column("Collection", justify="left")
    table.add_column("BodyPartsExamined", justify="left")
    table.add_column("Modalities", justify="left")
    table.add_column("Images", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("File Type", justify="left")
    table.add_column("Source", justify="left")
    collection_db = store.summary(update)

    for collection, info in collection_db.items():
        table.add_row(
            collection,
            ", ".join(info["BodyPartsExamined"]),
            ", ".join(info["Modalities"]),
            f"{info['Images']}",
            f"{info['Size']} GB",
            f"{info['File Type']}",
            f"{info['Source']}",
        )

    rprint(table)


@click.group("collections", no_args_is_help=True)
@click.help_option("--help", "-h")
def collections() -> None:
    """Inspect indexed collections."""


@collections.command("summary")
@click.help_option("--help", "-h")
@click.option(
    "--update",
    "-u",
    is_flag=True,
    help="Update the collections summary.",
)
def collections_summary(update: bool) -> None:
    """Display the collections summary."""
    store = IndexedDatasets()
    _print_collections_summary(store, update)


@collections.command("info")
@click.help_option("--help", "-h")
@click.argument("collection_name", type=str)
@click.option(
    "--tags",
    "-t",
    is_flag=True,
    help="Include supported query tags per modality (for building rules).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print metadata as JSON instead of a table.",
)
def collections_info(collection_name: str, tags: bool, as_json: bool) -> None:
    """Show basic metadata for a collection; use --tags for queryable columns."""
    store = IndexedDatasets()
    try:
        if as_json:
            payload = _build_collection_info_payload(
                store, collection_name, include_tags=tags
            )
            click.echo(json.dumps(payload, indent=2))
            return
        _print_collection_overview(store, collection_name)
        if tags:
            rprint()
            _print_supported_query_tags(store, collection_name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
