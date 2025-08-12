import click

from imgnet import ImgNet
from imgnet.query import ValidQuery
from imgnet.loggers import logger
from pathlib import Path
import json
from nbiatoolkit.nbia import NBIAClient

@click.command(no_args_is_help=True)
@click.argument(
    "output_path",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    )
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
@click.option(
    "--download",
    "-d",
    is_flag=True,
    help="Download queried datasets."
)
@click.option(
    "--username",
    "-u",
    type=str,
    default="nbia_guest",
    help="NBIA username."
)
@click.option(
    "--password",
    "-p",
    type=str,
    default="",
    help="NBIA password."
)
def query(
    output_path: Path,
    input_path: Path,
    collections: str | list[str],
    modalities: str | list[str],
    rules: str,
    download: bool,
    username: str,
    password: str
) -> None:
    """
    Query crawled TCIA datasets and optionally download selected DICOMs using NBIA Toolkit.
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
    download: `bool`
        If true, downloads the selected series' using NBIA Toolkit.
    username: `str`
        Username for accessing the NBIA API.
    password: `str`
        Password for accessing the NBIA API.

    Returns
    -------
    `None`
    
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    if download:
        try:
            (output_path/ "raw_data").mkdir(exist_ok=False)
        except FileExistsError as e:
            logger.error(f"{output_path/'raw_data'} already exists. Please remove the raw_data directory and try again.")
            raise e
        
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
        valid_query = ValidQuery(collections=collections, modalities=modalities, rules=rules)
        logger.info(f"Generated ValidQuery: \ncollections: {collections}\nmodalities: {modalities}\nrules: {rules}")
    
    with open(output_path / "valid_query_schema.json", "w") as f:
        json.dump(valid_query.model_json_schema(), f, indent=2)
    logger.info(f"Saved json schema to {output_path / 'valid_query_schema.json'}.")
    
    with open(output_path / "valid_query.json", "w") as f:
        json.dump(valid_query.model_dump(), f, indent=2)
    logger.info(f"Saved ValidQuery json to {output_path / 'valid_query.json'}.")
    
    client = NBIAClient(username=username, password=password)
    imgnet = ImgNet(output_path / "raw_data", client)
    results = imgnet.query(valid_query, download)

    results.to_csv(output_path / 'selected_dicoms.csv')
    logger.info(f"Saved selected SeriesUIDs to {output_path / 'selected_dicoms.csv'}.")
