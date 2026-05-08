import click
from pathlib import Path
import json

from imgnet.imgnet import ImgNet
from imgnet.query import ValidQuery, parse_rule_node
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
    "--collection",
    "-c",
    type=str,
    multiple=True,
    default = None,
    help = "Collection to query, or 'all' to query every collection (example: `-c 4D-Lung -c Adrenal-ACC-Ki67-Seg`)"
)
@click.option(
    "--modality",
    "-m",
    type=str,
    multiple=True,
    default=None,
    help="Modality to query (example: `-m CT -m RTSTRUCT -m SEG`)"
)
@click.option(
    "--file-type",
    "-ft",
    type=click.Choice(
        [ft.value for ft in FileType], case_sensitive=False
    ),
    default=None,
    help="Restrict query to a specific file type "
    "(e.g. 'dicom', 'nifti'). Default for no restriction.",
)
@click.option(
    "--rules",
    "-r",
    type=str,
    default=None,
    multiple=True,
    help="filter rules."
)


def query(
    output_dir: Path | None,
    input_path: Path | None,
    collection: list[str],
    modality: list[str],
    file_type: str | None,
    rules: list[str] | None,
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
        collections = list(collection) if collection else None
        modalities = list(modality) if modality else None
        parsed_rules = [parse_rule_node(rule) for rule in rules] if rules else None

        kwargs: dict[str, object] = {
            "collections": collections,
            "modalities": modalities,
            "rules": parsed_rules,
        }
        if file_type is not None:
            kwargs["file_type"] = file_type.lower()

        valid_query = ValidQuery(**kwargs)  # type: ignore[arg-type]
        logger.info(
            "Generated ValidQuery: \n"
            f"{valid_query.__repr__()}"
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
