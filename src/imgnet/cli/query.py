import click
from pathlib import Path
import json

from imgnet.imgnet import ImgNet
from imgnet.query import ValidQuery
from imgnet.collections.store import IndexedDatasets
from imgnet.loggers import logger

@click.command(no_args_is_help=True)
@click.option(
    "--output_path",
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
    "--input_path",
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
    "--rules",
    "-r",
    type=str,
    default=None,
    help="JSON string of filter rules."
)

def query(
    output_path: Path | None,
    input_path: Path | None,
    collections: str | list[str],
    modalities: str | list[str],
    rules: str,
) -> None:
    """
    Query crawled TCIA datasets and optionally download selected DICOMs using idc-index.
    A list of selected seriesUIDs for each collection will be saved at `output_path`/selected_seriesuids.csv,
    along with the JSON schema for a ValidQuery object and a JSON representation of the supplied query.

    Parameters
    ----------

    output_path: `Path`
        The directory where query results will be saved.
    input_path: `Path`
        A JSON file representing the query parameters.
    collections: `str` | `list[str]`
        The list of collections to query, if `input_path` is not supplied.
    modalities: `str` | `list[str]`
        The list of modality queries to run, if `input_path` is not supplied.
    rules: `str`
        A JSON string representation of the filter rules to apply to the query, if `input_path` is not supplied.
    """
    if output_path is not None:
        output_path = Path(output_path)
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
        valid_query = ValidQuery(collections=collections, modalities=modalities, rules=rules) # 
        logger.info(f"Generated ValidQuery: \ncollections: {collections}\nmodalities: {modalities}\nrules: {rules}\n")
    
    store = IndexedDatasets()
    imgnet = ImgNet(output_path, store=store)
    results = imgnet.query(valid_query)

    results.to_csv(output_path / 'query_results.csv')
    logger.info(f"Saved query results to {output_path / 'query_results.csv'}.")

    with open(output_path / "valid_query_schema.json", "w") as f:
        json.dump(valid_query.model_json_schema(), f, indent=2)
    logger.info(f"Saved json schema to {output_path / 'valid_query_schema.json'}.")
    
    with open(output_path / "valid_query.json", "w") as f:
        json.dump(valid_query.model_dump(), f, indent=2)
    logger.info(f"Saved ValidQuery json to {output_path / 'valid_query.json'}.")

    click.echo(str(output_path / "query_results.csv"))
