import click
from pathlib import Path
import json

from imgnet.imgnet import ImgNet
from imgnet.query import ValidQuery
from imgnet.collections.store import IndexedDatasets
from imgnet.collections.source import FileType
from imgnet.loggers import logger

@click.command(no_args_is_help=True)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    default=None,
    help="Path to the output directory."
)
@click.option(
    "--input-path",
    "-i",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    default=None,
    help="Path to a valid query json."
)
@click.option(
    "--collections",
    "-c",
    type=str,
    multiple=True,
    default = ["all"],
    help = "List of collections to query, or 'all' to query every collection (example: `-c collection1 collection2`)"
)
@click.option(
    "--modalities",
    "-m",
    type=str,
    multiple=True,
    default=['all'],
    help="List of modality queries (example: `-m CT,RTSTRUCT CT,SEG`)"
)
@click.option(
    "--file_type",
    "-ft",
    type=click.Choice(
        ["all"] + [ft.value for ft in FileType], case_sensitive=False
    ),
    default=None,
    help="Restrict query to a specific file type "
    "(e.g. 'dicom', 'nifti'). Use 'all' (default) for no restriction.",
)
@click.option(
    "--rules",
    "-r",
    type=str,
    default=None,
    help="JSON string of filter rules."
)


def query(
    output_dir: Path | None,
    input_path: Path | None,
    collections: str | list[str],
    modalities: str | list[str],
    file_type: str | None,
    rules: str,
) -> None:
    """Query indexed datasets and save selected series for downstream use.

    Saves query_results.csv, valid_query.json, and valid_query_schema.json
    to the output directory. Pipe the output into `imgnet download` to
    retrieve the matched series.
    """
    if output_dir is not None:
        output_path = Path(output_dir)
    else:
        output_path = Path.cwd() / "query_results"
        logger.warning(f"No output path provided, using: {output_path}")
    output_path.mkdir(exist_ok=True)

    if input_path:
        with open(input_path, 'rb') as f:
            valid_query = ValidQuery.model_validate_json(f.read())
        logger.info(f"Loaded ValidQuery from {input_path}.")
    else: 
        if rules is not None:
            rules = json.loads(rules)
        if len(collections) == 1:
            collections = collections[0]
        elif isinstance(collections, tuple):
            collections = list(collections)
        if isinstance(modalities, tuple):
            modalities = list(modalities)

        kwargs: dict[str, object] = {
            "collections": collections,
            "modalities": modalities,
            "rules": rules,
        }
        if file_type is not None:
            kwargs["file_type"] = file_type.lower()

        valid_query = ValidQuery(**kwargs)  # type: ignore[arg-type]
        logger.info(
            "Generated ValidQuery: \n"
            f"collections: {collections}\n"
            f"modalities: {modalities}\n"
            f"file_type: {valid_query.file_type}\n"
            f"rules: {rules}\n"
        )
    
    store = IndexedDatasets()
    imgnet = ImgNet(output_path, store=store)
    results = imgnet.query(valid_query)

    results.to_csv(output_path / 'query_results.csv')
    logger.info(f"Saved query results to {output_path / 'query_results.csv'}.")

    with open(output_path / "valid_query_schema.json", "w") as f:
        json.dump(valid_query.model_json_schema(), f, indent=2)
    logger.info(f"Saved json schema to {output_path / 'valid_query_schema.json'}.")
    
    with open(output_path / "valid_query.json", "w") as f:
        json.dump(valid_query.model_dump(mode="json"), f, indent=2)
    logger.info(f"Saved ValidQuery json to {output_path / 'valid_query.json'}.")

    click.echo(str(output_path / "query_results.csv"))
