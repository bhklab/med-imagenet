import click
from pathlib import Path
import json

from imgnet.imgnet import ImgNet
from imgnet.query import ValidQuery, parse_rule_node
from imgnet.collections.store import IndexedDatasets
from imgnet.collections.source import FileType
from imgnet.loggers import logger

@click.command(no_args_is_help=True)
@click.help_option("--help", "-h")
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
    help="Directory to save query results (default: ./query_results).",
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
    help="Re-run a previous query by loading a saved valid_query.json file.",
)
@click.option(
    "--collection",
    "-c",
    type=str,
    multiple=True,
    default=None,
    help="Include only files from this dataset collection. Can be specified multiple times. Use 'all' to query every collection.",
)
@click.option(
    "--modality",
    "-m",
    type=str,
    multiple=True,
    default=None,
    help="Include only files of this imaging modality (e.g., CT, MR, RTSTRUCT, SEG). Can be specified multiple times.",
)
@click.option(
    "--file-type",
    "-ft",
    type=click.Choice(
        [ft.value for ft in FileType], case_sensitive=False
    ),
    default=None,
    help="Include only files in this format: 'dicom' or 'nifti' (default: both).",
)
@click.option(
    "--rules",
    "-r",
    type=str,
    default=None,
    multiple=True,
    help="Filter series by metadata using boolean expressions. Format: modality(condition AND/OR condition). "
         "Example: -r 'CT(PatientAge>50 AND StudyDescription!=\"CHEST\")' "
         "Example: -r 'MR(SeriesDescription==\"T2_AXIAL\")' "
         "Example: -r 'CT(PatientSex==\"F\" OR PatientAge<30)'",
)


def query(
    output_dir: Path | None,
    input_path: Path | None,
    collection: list[str],
    modality: list[str],
    file_type: str | None,
    rules: list[str] | None,
) -> None:
    """Query medical imaging datasets and save matching DICOM series metadata.

    Searches across indexed imaging collections to find series that match your
    criteria. Results are saved as a CSV file that can be piped into 
    `imgnet download` to retrieve the actual image files.

    Output files in <output-dir>:
      - query_results.csv      : List of matching series with their metadata
      - valid_query.json       : The query used (can be reused with --input-path)
      - valid_query_schema.json: JSON schema for the query format

    \b
    QUERY LOGIC:
      All specified filters are combined with AND logic. For example, using
      --collection 4D-Lung --modality CT --rules "CT(PatientAge>50)"
      will only match CT series from the 4D-Lung collection where patient age > 50.

      Rules provide the most powerful filtering and support:
      - Operators: ==, !=, <=, >=, <, >
      - Logical: AND, OR, and parentheses for grouping
      - Values: strings (quoted), numbers, or unquoted words

    \b
    EXAMPLES:
      # Query all CT scans from the 4D-Lung collection
      imgnet query -c 4D-Lung -m CT

      # Query multiple collections for MR and CT, save to custom directory
      imgnet query -o ./my_query -c 4D-Lung -c Adrenal-ACC-Ki67-Seg -m MR -m CT

      # Find female patients over 50 with CT chest scans
      imgnet query -r 'CT(PatientSex=="F" AND PatientAge>50 AND StudyDescription=="CHEST")'

      # Find either CT or MR studies for patients under 30
      imgnet query -r 'CT(PatientAge<30) OR MR(PatientAge<30)'

      # Re-run a previous query
      imgnet query -i ./previous_query/valid_query.json

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
