import json
from collections import Counter
from textwrap import fill

import click
from rich import print as rprint
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console(force_terminal=True, color_system="truecolor")

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


def _aggregate_index_stats(collection_db: dict) -> dict:
    """Compute aggregate statistics across all collections."""
    total_images = 0
    total_size = 0.0
    modality_images: Counter[str] = Counter()
    modality_collections: Counter[str] = Counter()
    bodypart_images: Counter[str] = Counter()
    bodypart_collections: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    filetype_counts: Counter[str] = Counter()
    source_images: Counter[str] = Counter()
    filetype_images: Counter[str] = Counter()

    for _name, info in collection_db.items():
        images = info.get("Images", 0)
        total_images += images
        total_size += info.get("Size", 0.0)

        source = info.get("Source", "UNKNOWN")
        source_counts[source] += 1
        source_images[source] += images

        ftype = info.get("File Type", "UNKNOWN")
        filetype_counts[ftype] += 1
        filetype_images[ftype] += images

        for mod in info.get("Modalities", []):
            modality_images[mod] += images
            modality_collections[mod] += 1
        for bp in info.get("BodyPartsExamined", []):
            bodypart_images[bp] += images
            bodypart_collections[bp] += 1

    return {
        "total_collections": len(collection_db),
        "total_images": total_images,
        "total_size": total_size,
        "modality_images": modality_images,
        "modality_collections": modality_collections,
        "bodypart_images": bodypart_images,
        "bodypart_collections": bodypart_collections,
        "source_counts": source_counts,
        "source_images": source_images,
        "filetype_counts": filetype_counts,
        "filetype_images": filetype_images,
    }


_BAR_CHAR = "█"
_BAR_MAX_WIDTH = 30


def _bar(value: int, max_value: int, color: str = "green") -> Text:
    """Render a proportional horizontal bar using block characters."""
    if max_value == 0:
        return Text("")
    width = max(1, round(value / max_value * _BAR_MAX_WIDTH))
    bar = Text(_BAR_CHAR * width, style=color)
    bar.append(f" {value:,}", style="bold")
    return bar


def _distribution_table(
    title: str,
    counter: Counter,
    *,
    color: str = "green",
    top_n: int = 15,
    label_header: str = "Category",
    value_header: str = "Images",
) -> Table:
    """Build a Rich table with horizontal bar chart rows."""
    table = Table(
        title=title, show_header=True, header_style="bold", expand=False
    )
    table.add_column(label_header, justify="left", style="cyan", min_width=18)
    table.add_column(value_header, justify="left", min_width=38)

    items = counter.most_common(top_n)
    max_val = items[0][1] if items else 1
    for label, count in items:
        table.add_row(label or "(empty)", _bar(count, max_val, color))

    remaining = len(counter) - top_n
    if remaining > 0:
        table.add_row(f"... +{remaining} more", Text("", style="dim"))
    return table


def _render_overview(collection_db: dict) -> None:
    """Render the full index overview to the terminal."""
    stats = _aggregate_index_stats(collection_db)
    out = _console

    header = Table.grid(padding=(0, 4))
    header.add_column(justify="center")
    header.add_column(justify="center")
    header.add_column(justify="center")
    header.add_row(
        Text(f"{stats['total_collections']}", style="bold cyan")
        + Text(" collections"),
        Text(f"{stats['total_images']:,}", style="bold cyan")
        + Text(" images"),
        Text(f"{stats['total_size']:,.1f} GB", style="bold cyan")
        + Text(" total size"),
    )
    out.print(Align.center(Panel(header, title="[bold]Med-ImageNet Index Overview[/bold]")))

    src_table = Table(
        title="Sources", show_header=True, header_style="bold", expand=False
    )
    src_table.add_column("Source", style="cyan")
    src_table.add_column("Collections", justify="right")
    src_table.add_column("Images", justify="right")
    for src, cnt in stats["source_counts"].most_common():
        src_table.add_row(src, str(cnt), f"{stats['source_images'][src]:,}")

    ft_table = Table(
        title="File Types",
        show_header=True,
        header_style="bold",
        expand=False,
    )
    ft_table.add_column("Type", style="cyan")
    ft_table.add_column("Collections", justify="right")
    ft_table.add_column("Images", justify="right")
    for ft, cnt in stats["filetype_counts"].most_common():
        ft_table.add_row(ft, str(cnt), f"{stats['filetype_images'][ft]:,}")

    out.print(Align.center(Columns([src_table, ft_table], padding=(0, 4))))

    out.print(Align.center(
        _distribution_table(
            "Images by Modality",
            stats["modality_images"],
            color="magenta",
            top_n=10,
            label_header="Modality",
        )
    ))

    sorted_by_size = sorted(
        collection_db.items(),
        key=lambda kv: kv[1].get("Size", 0),
        reverse=True,
    )
    size_table = Table(
        title="Largest Collections (by size)",
        show_header=True,
        header_style="bold",
    )
    size_table.add_column("Collection", style="cyan")
    size_table.add_column("Size (GB)", justify="right")
    size_table.add_column("Images", justify="right")
    size_table.add_column("Modalities", justify="left")
    for name, info in sorted_by_size[:5]:
        size_table.add_row(
            name,
            f"{info.get('Size', 0):,.2f}",
            f"{info['Images']:,}",
            ", ".join(info.get("Modalities", [])) or "—",
        )
    out.print(Align.center(size_table))


@click.group("collections", no_args_is_help=True)
@click.help_option("--help", "-h")
def collections() -> None:
    """Explore datasets available in the ImgNet index.

    Collections are indexed medical imaging datasets that can be searched,
    inspected, and downloaded using ImgNet commands.
    """


@collections.command("summary")
@click.help_option("--help", "-h")
@click.option(
    "--update",
    "-u",
    is_flag=True,
    help="Rebuild the collections summary before generating the overview."
)


def collections_summary(update: bool) -> None:
    """Display a summary of all indexed collections.

    Shows basic information including collection size, image count,
    modalities, file formats, and data sources.

    Use --update to rebuild the summary from the collection metadata.
    """
    store = IndexedDatasets()
    _print_collections_summary(store, update)


@collections.command("info")
@click.help_option("--help", "-h")
@click.argument("collection_name", type=str)
@click.option(
    "--tags",
    "-t",
    is_flag=True,
    help="Show metadata fields available for querying, grouped by modality.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output collection metadata as JSON instead of a formatted table.",
)
def collections_info(collection_name: str, tags: bool, as_json: bool) -> None:
    """Display detailed information about an indexed collection.

    Shows metadata such as the data source, file format, image count,
    modalities, body regions, and collection description.

    Use --tags to display the metadata fields available for filtering
    when building ImgNet queries.
    """
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


@collections.command("overview")
@click.help_option("--help", "-h")
@click.option(
    "--update",
    "-u",
    is_flag=True,
    help="Rebuild the collections summary before generating the overview.",
)
def collections_overview(update: bool) -> None:
    """Display a visual overview of the ImgNet dataset index.

    Shows aggregate statistics across all collections, including:
    number of collections, total images, storage size, data sources,
    file formats, modality distributions, and largest collections.
    """
    store = IndexedDatasets()
    collection_db = store.summary(update)
    _render_overview(collection_db)
