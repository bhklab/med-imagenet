"""Build an nnU-Net v2 raw dataset from med-imagenet query results."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from imgnet.loggers import logger, tqdm_logging_redirect
from imgnet.nnunet.images import (
    CaseResult,
    WriteOptions,
    image_path,
    init_worker,
    label_path,
    write_case,
)
from imgnet.nnunet.plan import (
    TEST,
    TRAIN,
    Case,
    Plan,
    PlanError,
    Skipped,
    plan_dataset,
    read_query_results,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from imgnet.nnunet.config import NnUNetConfig

MANIFEST_NAME = "imgnet_manifest.csv"
_SUBDIRS = ("imagesTr", "labelsTr", "imagesTs", "labelsTs")


class OutputExistsError(FileExistsError):
    """The dataset folder (or its ID) is already in use."""


@dataclass
class Failure:
    """A case that was planned but could not be written."""

    case: Case
    error: str


@dataclass
class BuildResult:
    """Summary of a build (or a dry run)."""

    dataset_dir: Path
    plan: Plan
    written: list[Case] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    dry_run: bool = False

    @property
    def num_train(self) -> int:
        """Training cases written (or planned, for a dry run)."""
        cases = self.plan.cases if self.dry_run else self.written
        return sum(c.split == TRAIN for c in cases)

    @property
    def num_test(self) -> int:
        """Test cases written (or planned, for a dry run)."""
        cases = self.plan.cases if self.dry_run else self.written
        return sum(c.split == TEST for c in cases)


def build_dataset(
    config: NnUNetConfig,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    num_workers: int | None = None,
) -> BuildResult:
    """Plan and write an nnU-Net v2 raw dataset.

    Parameters
    ----------
    config
        Validated config, see :func:`imgnet.nnunet.config.load_config`.
    dry_run
        Only plan and report; write nothing.
    overwrite
        Delete an existing dataset folder of the same name first.
    num_workers
        Worker processes; overrides ``config.num_workers``.

    Returns
    -------
    BuildResult
        What was written, skipped and failed. ``dataset.json`` only counts
        written cases, so failed cases never break nnU-Net's checks.
    """
    if not config.input_dir.is_dir():
        msg = f"input_dir does not exist: {config.input_dir}"
        raise FileNotFoundError(msg)

    df = read_query_results(config.query_results_csv)
    plan = plan_dataset(df, config)
    result = BuildResult(config.dataset_dir, plan, dry_run=dry_run)
    _log_plan(plan)
    if dry_run:
        return result
    if not plan.cases:
        msg = "No cases planned; every sample was skipped."
        raise PlanError(msg)
    if not any(c.split == TRAIN for c in plan.cases):
        msg = "No training cases planned; nnU-Net needs at least one."
        raise PlanError(msg)

    _prepare_output(config, overwrite=overwrite)
    options = WriteOptions(
        dataset_dir=config.dataset_dir,
        file_ending=config.file_ending,
        scan_link_mode=config.scan_link_mode,
        geometry_tolerance=config.geometry_tolerance,
    )
    workers = num_workers or config.num_workers or available_cpus()
    workers = max(1, min(workers, len(plan.cases)))
    logger.info(
        "Writing %d cases to %s with %d worker(s).",
        len(plan.cases),
        config.dataset_dir,
        workers,
    )

    by_id = {c.case_id: c for c in plan.cases}
    with tqdm_logging_redirect():
        for case_id, outcome in _run(plan.cases, options, workers):
            case = by_id[case_id]
            if isinstance(outcome, CaseResult):
                result.written.append(case)
                for warning in outcome.warnings:
                    logger.warning(warning)
            else:
                result.failures.append(Failure(case, outcome))
                logger.error("Failed %s: %s", case_id, outcome)

    _write_dataset_json(config, result.written)
    _write_manifest(config.dataset_dir, result)
    _verify_counts(config, result)
    logger.info(
        "Wrote %s: train=%d test=%d skipped=%d failed=%d",
        config.dataset_dir,
        result.num_train,
        result.num_test,
        len(plan.skipped),
        len(result.failures),
    )
    return result


def available_cpus() -> int:
    """CPUs this process may use (respects SLURM/cgroup limits on Linux)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Running cases
# ---------------------------------------------------------------------------


def _run(
    cases: list[Case], options: WriteOptions, workers: int
) -> Iterator[tuple[str, CaseResult | str]]:
    """Yield (case_id, result or error message) as cases finish."""
    progress = tqdm(total=len(cases), desc="cases", unit="case")
    try:
        if workers == 1:
            init_worker()
            for case in cases:
                yield case.case_id, _attempt(case, options)
                progress.update()
            return
        # spawn: safe with the threads pandas/pyarrow may have started.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=init_worker
        ) as pool:
            futures = {
                pool.submit(write_case, case, options): case.case_id
                for case in cases
            }
            for future in as_completed(futures):
                try:
                    outcome: CaseResult | str = future.result()
                except Exception as e:  # noqa: BLE001 - report every failure
                    outcome = _describe(e)
                yield futures[future], outcome
                progress.update()
    finally:
        progress.close()


def _attempt(case: Case, options: WriteOptions) -> CaseResult | str:
    try:
        return write_case(case, options)
    except Exception as e:  # noqa: BLE001 - report every failure
        return _describe(e)


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------


def _prepare_output(config: NnUNetConfig, *, overwrite: bool) -> None:
    dataset_dir = config.dataset_dir
    prefix = f"Dataset{config.dataset_id:03d}_"
    if config.nnunet_raw_dir.is_dir():
        others = sorted(
            p.name
            for p in config.nnunet_raw_dir.iterdir()
            if p.name.startswith(prefix) and p.name != dataset_dir.name
        )
        if others:
            msg = (
                f"Dataset ID {config.dataset_id} is already used by {others} "
                f"in {config.nnunet_raw_dir}; nnU-Net needs IDs to be unique."
            )
            raise OutputExistsError(msg)

    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        if not overwrite:
            msg = (
                f"{dataset_dir} already exists and is not empty. Leftover "
                "cases would end up in the dataset; use --overwrite to "
                "replace it."
            )
            raise OutputExistsError(msg)
        logger.warning("Deleting existing dataset folder %s", dataset_dir)
        shutil.rmtree(dataset_dir)

    for sub in _SUBDIRS:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)


def _write_dataset_json(config: NnUNetConfig, written: list[Case]) -> None:
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
    path.write_text(json.dumps(dataset, indent=2) + "\n")


def _write_manifest(dataset_dir: Path, result: BuildResult) -> None:
    """One row per sample: where it went, or why it didn't."""
    failed = {f.case.case_id: f.error for f in result.failures}
    rows = [
        _manifest_row(
            c,
            "failed" if c.case_id in failed else "written",
            failed.get(c.case_id, ""),
        )
        for c in result.plan.cases
    ]
    rows += [_skipped_row(s) for s in result.plan.skipped]
    columns = [
        "collection", "sample_id", "case_id", "split", "status", "reason",
        "scan", "masks",
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.sort_values(["collection", "sample_id"], kind="stable")
    frame.to_csv(dataset_dir / MANIFEST_NAME, index=False)


def _manifest_row(case: Case, status: str, reason: str) -> dict[str, str]:
    return {
        "collection": case.collection,
        "sample_id": case.sample_id,
        "case_id": case.case_id,
        "split": case.split,
        "status": status,
        "reason": reason,
        "scan": str(case.scan),
        "masks": ";".join(str(p) for p in case.masks),
    }


def _skipped_row(skipped: Skipped) -> dict[str, str]:
    return {
        "collection": skipped.collection,
        "sample_id": skipped.sample_id,
        "status": "skipped",
        "reason": skipped.reason,
    }


def _verify_counts(config: NnUNetConfig, result: BuildResult) -> None:
    """Every written case must have exactly its two files on disk."""
    missing = [
        str(p)
        for case in result.written
        for p in (
            image_path(config.dataset_dir, case, config.file_ending),
            label_path(config.dataset_dir, case, config.file_ending),
        )
        if not (p.is_file() or p.is_symlink())
    ]
    if missing:
        msg = f"Written cases are missing files: {missing[:5]}"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _log_plan(plan: Plan) -> None:
    splits = Counter((c.collection, c.split) for c in plan.cases)
    collections = sorted(
        {c.collection for c in plan.cases}
        | {s.collection for s in plan.skipped}
    )
    for name in collections:
        reasons = Counter(
            _reason_kind(s.reason)
            for s in plan.skipped
            if s.collection == name
        )
        logger.info(
            "%s: train=%d test=%d skipped=%d %s",
            name,
            splits[(name, TRAIN)],
            splits[(name, TEST)],
            sum(reasons.values()),
            dict(reasons) if reasons else "",
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
