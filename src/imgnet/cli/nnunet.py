from pathlib import Path

import click

from imgnet.loggers import logger


@click.command(no_args_is_help=True)
@click.help_option("--help", "-h")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Select cases and run every check that does not write, then report counts. Writes nothing.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Delete the existing dataset folder of the same name first.",
)
@click.option(
    "--num-workers",
    "-j",
    type=click.IntRange(min=1),
    default=None,
    help="Worker processes (default: num_workers from the config, else all available CPUs).",
)
def nnunet(
    config_path: Path,
    dry_run: bool,
    overwrite: bool,
    num_workers: int | None,
) -> None:
    """Build an nnU-Net v2 raw dataset from downloaded NIfTI collections.

    Reads CONFIG_PATH (JSON), picks one scan and its masks for every sample
    in the query results, makes a train/test split and writes
    <nnunet_raw_dir>/DatasetXXX_<name>/ with imagesTr, labelsTr, imagesTs,
    labelsTs, dataset.json and imgnet_manifest.csv (one row per sample,
    including skipped and failed ones).

    \b
    MINIMAL CONFIG:
      {
        "input_dir": "imgnet_output/srcdata",
        "query_results_csv": "query_results/query_results.csv",
        "nnunet_raw_dir": "nnUNet_raw",
        "dataset_id": 1,
        "dataset_name": "Lesions",
        "labels": {"background": 0, "liver tumor": 1},
        "collections": [
          {"collection": "Abdomen Atlas 3", "method": "combine",
           "intensity_map": {"liver_lesion": 1}},
          {"collection": "MSWAL", "method": "remap",
           "intensity_map": {"3": 1}}
        ]
      }

    \b
    EXAMPLES:
      # Check what would be written
      imgnet nnunet config.json --dry-run

      # Build with 16 worker processes
      imgnet nnunet config.json -j 16
    """
    from imgnet.nnunet.dataset_builder import build_nnunet_dataset
    from imgnet.nnunet.dataset_config import load_dataset_config
    from imgnet.nnunet.errors import NnUNetBuildError

    try:
        config = load_dataset_config(config_path)
        result = build_nnunet_dataset(
            config,
            dry_run=dry_run,
            overwrite=overwrite,
            num_workers=num_workers,
        )
    except NnUNetBuildError as e:
        raise click.ClickException(str(e)) from e

    summary = (
        f"train={result.num_train} test={result.num_test} "
        f"skipped={len(result.selection.skipped)}"
    )
    if dry_run:
        click.echo(f"Dry run, nothing written. Would write {summary} to {result.dataset_dir}")
        return

    click.echo(f"{result.dataset_dir}: {summary} failed={len(result.failures)}")
    if result.failures:
        logger.error(
            "%d cases failed and were left out of the dataset; see "
            "imgnet_manifest.csv.",
            len(result.failures),
        )
        raise SystemExit(1)
