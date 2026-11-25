"""Build an nnU-Net v2 raw dataset from med-imagenet query results."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.nnunet.case_selection import (
    TEST,
    TRAIN,
    CaseSelection,
    SelectedCase,
    SkippedSample,
    read_query_results,
    select_cases,
)
from imgnet.nnunet.case_writer import (
    CaseJob,
    CaseWriteResult,
    WriteOptions,
    image_path,
    init_worker,
    label_path,
    remove_case_outputs,
    write_case,
)
from imgnet.nnunet.errors import (
    DatasetExistsError,
    DatasetOutputError,
    NnUNetBuildError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from imgnet.nnunet.dataset_config import NnUNetDatasetConfig

MANIFEST_NAME = "imgnet_manifest.csv"
_SUBDIRS = ("imagesTr", "labelsTr", "imagesTs", "labelsTs")
_MAX_LOGGED_FAILURES = 20
_MAX_EXAMPLES = 5
WORKER_DIED = (
    "not written: a worker process died (often out of memory). Rerun with "
    "a lower --num-workers and --overwrite."
)


@dataclass
class FailedCase:
    """A case that was selected but could not be written."""

    case: SelectedCase
    error: str


@dataclass
class DatasetBuildResult:
    """Summary of a build (or a dry run)."""

    dataset_dir: Path
    selection: CaseSelection
    written: list[SelectedCase] = field(default_factory=list)
    failures: list[FailedCase] = field(default_factory=list)
    # case_id -> note for the manifest, e.g. unmapped mask values.
    notes: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def num_train(self) -> int:
        """Training cases written (or selected, for a dry run)."""
        cases = self.selection.cases if self.dry_run else self.written
        return sum(c.split == TRAIN for c in cases)

    @property
    def num_test(self) -> int:
        """Test cases written (or selected, for a dry run)."""
        cases = self.selection.cases if self.dry_run else self.written
        return sum(c.split == TEST for c in cases)


def build_nnunet_dataset(
    config: NnUNetDatasetConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    num_workers: int | None = None,
) -> DatasetBuildResult:
    """Select cases and write an nnU-Net v2 raw dataset.

    Parameters
    ----------
    config
        Validated config, see :func:`imgnet.nnunet.dataset_config.load_dataset_config`.
    dry_run
        Select cases and run every check that doesn't write, but write
        nothing.
    overwrite
        Delete an existing dataset folder of the same name first.
    num_workers
        Worker processes; overrides ``config.num_workers``.

    Returns
    -------
    DatasetBuildResult
        What was written, skipped and failed.

    Raises
    ------
    NnUNetBuildError
        For every expected failure: bad input, case selection problems,
        an output folder that is in use, or output that fails verification.
    """
    if not config.input_dir.is_dir():
        msg = f"input_dir does not exist: {config.input_dir}"
        raise NnUNetBuildError(msg)

    df = read_query_results(config.query_results_csv)
    selection = select_cases(df, config)
    _log_selection(selection)
    _check_output(config, overwrite=overwrite)
    result = DatasetBuildResult(config.dataset_dir, selection, dry_run=dry_run)
    if dry_run:
        if overwrite and config.dataset_dir.exists():
            logger.warning(
                "Dry run: the real run would delete %s first.",
                config.dataset_dir,
            )
        return result

    _reset_output(config)
    options = WriteOptions(
        dataset_dir=config.dataset_dir,
        file_ending=config.file_ending,
        scan_link_mode=config.scan_link_mode,
        geometry_tolerance=config.geometry_tolerance,
    )
    workers = num_workers or config.num_workers or available_cpus()
    workers = max(1, min(workers, len(selection.cases)))
    logger.info(
        "Writing %d cases to %s with %d worker(s).",
        len(selection.cases),
        config.dataset_dir,
        workers,
    )
    _write_cases(selection.cases, options, workers, result)
    _log_failures(result.failures)

    try:
        _write_dataset_json(config, result.written)
        _write_manifest(config.dataset_dir, result)
    except OSError as e:
        msg = f"Could not write dataset files in {config.dataset_dir}: {e}"
        raise DatasetOutputError(msg) from e
    verify_dataset_files(config, result.written)
    logger.info(
        "Wrote %s: train=%d test=%d skipped=%d failed=%d",
        config.dataset_dir,
        result.num_train,
        result.num_test,
        len(selection.skipped),
        len(result.failures),
    )
    return result


def available_cpus() -> int:
    """CPUs this process may run on.

    Respects CPU affinity on Linux (e.g. SLURM cpusets), but not CPU quotas.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Running cases
# ---------------------------------------------------------------------------


def to_job(case: SelectedCase) -> CaseJob:
    """The worker payload for *case* (no pandas or logging needed to load)."""
    return CaseJob(
        case_id=case.case_id,
        split=case.split,
        method=case.method,
        scan=case.scan,
        masks=tuple(case.masks),
        labels=tuple(case.labels),
        remap=tuple(case.remap.items()),
    )


def _write_cases(
    cases: list[SelectedCase],
    options: WriteOptions,
    workers: int,
    result: DatasetBuildResult,
) -> None:
    jobs = {c.case_id: to_job(c) for c in cases}
    by_id = {c.case_id: c for c in cases}
    unmapped: dict[str, Counter[float]] = {}
    unmapped_cases: Counter[str] = Counter()
    with tqdm_logging_redirect():
        for case_id, outcome in _run(list(jobs.values()), options, workers):
            case = by_id[case_id]
            if isinstance(outcome, str):
                # The worker may have died mid-write; clean up here too.
                remove_case_outputs(jobs[case_id], options)
                result.failures.append(FailedCase(case, outcome))
                continue
            result.written.append(case)
            if outcome.num_unmapped_values:
                unmapped_cases[case.collection] += 1
                unmapped.setdefault(case.collection, Counter()).update(
                    outcome.unmapped_values
                )
                result.notes[case_id] = _unmapped_note(outcome)
    _log_unmapped(unmapped, unmapped_cases)


def _run(
    jobs: list[CaseJob], options: WriteOptions, workers: int
) -> Iterator[tuple[str, CaseWriteResult | str]]:
    """Yield (case_id, result or error message) as cases finish."""
    progress = tqdm(total=len(jobs), desc="cases", unit="case")
    try:
        if workers == 1:
            init_worker()
            for job in jobs:
                yield job.case_id, _attempt(job, options)
                progress.update()
            return
        # spawn: safe with the threads pandas/pyarrow may have started.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=init_worker
        ) as pool:
            futures = {
                pool.submit(write_case, job, options): job.case_id
                for job in jobs
            }
            for future in as_completed(futures):
                try:
                    outcome: CaseWriteResult | str = future.result()
                except BrokenProcessPool:
                    outcome = WORKER_DIED
                except Exception as e:  # noqa: BLE001 - report every failure
                    outcome = _describe(e)
                yield futures[future], outcome
                progress.update()
    finally:
        progress.close()


def _attempt(job: CaseJob, options: WriteOptions) -> CaseWriteResult | str:
    try:
        return write_case(job, options)
    except Exception as e:  # noqa: BLE001 - report every failure
        return _describe(e)


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------


def _check_output(config: NnUNetDatasetConfig, *, overwrite: bool) -> None:
    """Read-only checks on the output location (also run for dry runs)."""
    dataset_dir = config.dataset_dir
    prefix = f"Dataset{config.dataset_id:03d}_"
    try:
        if config.nnunet_raw_dir.is_dir():
            others = sorted(
                p.name
                for p in config.nnunet_raw_dir.iterdir()
                if p.name.startswith(prefix) and p.name != dataset_dir.name
            )
            if others:
                msg = (
                    f"Dataset ID {config.dataset_id} is already used by "
                    f"{others} in {config.nnunet_raw_dir}; nnU-Net needs IDs "
                    "to be unique."
                )
                raise DatasetExistsError(msg)
        in_use = dataset_dir.exists() and any(dataset_dir.iterdir())
    except OSError as e:
        if isinstance(e, NnUNetBuildError):
            raise
        msg = f"Could not inspect {config.nnunet_raw_dir}: {e}"
        raise DatasetOutputError(msg) from e
    if in_use and not overwrite:
        msg = (
            f"{dataset_dir} already exists and is not empty. Leftover cases "
            "would end up in the dataset; use --overwrite to replace it."
        )
        raise DatasetExistsError(msg)


def _reset_output(config: NnUNetDatasetConfig) -> None:
    """Delete any previous dataset folder and create the empty layout."""
    dataset_dir = config.dataset_dir
    try:
        if dataset_dir.exists():
            logger.warning("Deleting existing dataset folder %s", dataset_dir)
            shutil.rmtree(dataset_dir)
        for sub in _SUBDIRS:
            (dataset_dir / sub).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Could not prepare {dataset_dir}: {e}"
        raise DatasetOutputError(msg) from e


def verify_dataset_files(
    config: NnUNetDatasetConfig, written: list[SelectedCase]
) -> None:
    """The output folders must hold exactly the written cases' files.

    Lists each folder the way nnU-Net does (every file ending in
    ``file_ending``), so leftovers, orphans and case-insensitive name
    clashes are all caught.

    Raises
    ------
    DatasetOutputError
        If any folder has unexpected or missing files.
    """
    ending = config.file_ending
    expected: dict[str, set[str]] = {sub: set() for sub in _SUBDIRS}
    for case in written:
        for path in (
            image_path(config.dataset_dir, case.case_id, case.split, ending),
            label_path(config.dataset_dir, case.case_id, case.split, ending),
        ):
            expected[path.parent.name].add(path.name)
    problems: list[str] = []
    for sub, names in expected.items():
        folder = config.dataset_dir / sub
        actual = {p.name for p in folder.iterdir() if p.name.endswith(ending)}
        unexpected = sorted(actual - names)[:_MAX_EXAMPLES]
        missing = sorted(names - actual)[:_MAX_EXAMPLES]
        if unexpected:
            problems.append(f"{sub} has unexpected files {unexpected}")
        if missing:
            problems.append(f"{sub} is missing {missing}")
    if problems:
        msg = (
            f"{config.dataset_dir} does not match the cases in dataset.json: "
            + "; ".join(problems)
        )
        raise DatasetOutputError(msg)


def _write_dataset_json(
    config: NnUNetDatasetConfig, written: list[SelectedCase]
) -> None:
    dataset = {
        "name": config.dataset_folder_name,
        "description": (
            f"{config.dataset_folder_name}, built by imgnet from "
            + ", ".join(c.collection for c in config.collections)
        ),
        "channel_names": {"0": config.channel_name},
        "labels": dict(sorted(config.labels.items(), key=lambda kv: kv[1])),
        "numTraining": sum(c.split == TRAIN for c in written),
        "file_ending": config.file_ending,
        # Not used by nnU-Net; for reference.
        "numTest": sum(c.split == TEST for c in written),
    }
    path = config.dataset_dir / "dataset.json"
    path.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")


def _write_manifest(dataset_dir: Path, result: DatasetBuildResult) -> None:
    """One row per sample: where it went, or why it didn't."""
    failed = {f.case.case_id: f.error for f in result.failures}
    rows = [
        _manifest_row(
            c,
            "failed" if c.case_id in failed else "written",
            failed.get(c.case_id, ""),
            result.notes.get(c.case_id, ""),
        )
        for c in result.selection.cases
    ]
    rows += [_skipped_row(s) for s in result.selection.skipped]
    columns = [
        "collection", "sample_id", "case_id", "split", "status", "reason",
        "notes", "scan", "masks",
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.sort_values(["collection", "sample_id"], kind="stable")
    frame.to_csv(dataset_dir / MANIFEST_NAME, index=False)


def _manifest_row(
    case: SelectedCase, status: str, reason: str, notes: str
) -> dict[str, str]:
    return {
        "collection": case.collection,
        "sample_id": case.sample_id,
        "case_id": case.case_id,
        "split": case.split,
        "status": status,
        "reason": reason,
        "notes": notes,
        "scan": str(case.scan),
        "masks": ";".join(str(p) for p in case.masks),
    }


def _skipped_row(skipped: SkippedSample) -> dict[str, str]:
    return {
        "collection": skipped.collection,
        "sample_id": skipped.sample_id,
        "status": "skipped",
        "reason": skipped.reason,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_selection(selection: CaseSelection) -> None:
    splits = Counter((c.collection, c.split) for c in selection.cases)
    collections = sorted(
        {c.collection for c in selection.cases}
        | {s.collection for s in selection.skipped}
    )
    for name in collections:
        reasons = Counter(
            _reason_kind(s.reason)
            for s in selection.skipped
            if s.collection == name
        )
        logger.info(
            "%s: train=%d test=%d skipped=%d",
            name,
            splits[(name, TRAIN)],
            splits[(name, TEST)],
            sum(reasons.values()),
        )
        if reasons:
            logger.warning(
                "%s: %d samples skipped (%s). Per-sample reasons are in %s.",
                name,
                sum(reasons.values()),
                ", ".join(f"{k}: {v}" for k, v in reasons.items()),
                MANIFEST_NAME,
            )


def _reason_kind(reason: str) -> str:
    """Group reasons like 'file not found: /a/b' under 'file not found'."""
    return reason.split(":", 1)[0].split(" [", 1)[0]


def _log_failures(failures: list[FailedCase]) -> None:
    died = [f for f in failures if f.error == WORKER_DIED]
    other = [f for f in failures if f.error != WORKER_DIED]
    for failure in other[:_MAX_LOGGED_FAILURES]:
        logger.error("Failed %s: %s", failure.case.case_id, failure.error)
    if len(other) > _MAX_LOGGED_FAILURES:
        logger.error(
            "... and %d more failed cases; see %s.",
            len(other) - _MAX_LOGGED_FAILURES,
            MANIFEST_NAME,
        )
    if died:
        logger.error(
            "%d cases were not written because a worker process died "
            "(often out of memory). Rerun with a lower --num-workers and "
            "--overwrite.",
            len(died),
        )


def _unmapped_note(outcome: CaseWriteResult) -> str:
    values = outcome.unmapped_values
    more = outcome.num_unmapped_values - len(values)
    suffix = f" (+{more} more)" if more > 0 else ""
    return f"mask values not in intensity_map set to 0: {values}{suffix}"


def _log_unmapped(
    unmapped: dict[str, Counter[float]], unmapped_cases: Counter[str]
) -> None:
    for collection, values in sorted(unmapped.items()):
        common = values.most_common(10)
        shown = ", ".join(f"{v} ({n} cases)" for v, n in common)
        more = len(values) - len(common)
        logger.warning(
            "%s: %d cases have mask values that are not in intensity_map "
            "and were set to 0: %s%s. Per-case values are in the notes "
            "column of %s.",
            collection,
            unmapped_cases[collection],
            shown,
            f", +{more} more" if more > 0 else "",
            MANIFEST_NAME,
        )
