"""Turn query results into a list of nnU-Net cases (no image I/O).

Case selection decides, for every sample, which scan and which masks to use, which
split it lands in and what its case ID is. All ambiguity is reported here,
before anything is written.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from imgnet.loggers import logger
from imgnet.nnunet.dataset_config import Regex
from imgnet.nnunet.errors import CaseSelectionError

if TYPE_CHECKING:
    from pathlib import Path

    from imgnet.nnunet.dataset_config import (
        CollectionConfig,
        NnUNetDatasetConfig,
        Selector,
    )

REQUIRED_COLUMNS = ("Collection", "SampleID", "file_type", "filepath")
TRAIN, TEST = "Tr", "Ts"
_MAX_EXAMPLES = 5
_INVALID_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_TRAILING_ZERO_DECIMALS = re.compile(r"\.0+$")
# Tolerance for float error in test_size * n_groups, e.g. 0.07 * 100.
_SPLIT_EPSILON = 1e-9


@dataclass
class SelectedCase:
    """One nnU-Net case: a scan plus the masks that make its label."""

    collection: str
    sample_id: str
    scan: Path
    method: str
    # combine: one binary mask per ROI; remap: exactly one mask.
    masks: list[Path]
    # combine: output label for each mask, in config order (later wins).
    labels: list[int] = field(default_factory=list)
    # remap: source value -> output label.
    remap: dict[int, int] = field(default_factory=dict)
    case_id: str = ""
    split: str = ""
    # Modality column of the chosen scan row, if the CSV has one.
    modality: str | None = None


@dataclass
class SkippedSample:
    """A sample that was left out, and why."""

    collection: str
    sample_id: str
    reason: str


@dataclass
class CaseSelection:
    """Result of case selection: cases to write and samples left out."""

    cases: list[SelectedCase]
    skipped: list[SkippedSample]


def read_query_results(path: Path) -> pd.DataFrame:
    """Read a query results CSV with every column as text."""
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    # ValueError covers pandas' EmptyDataError/ParserError and
    # UnicodeDecodeError.
    except (OSError, ValueError) as e:
        msg = f"Could not read query results {path}: {e}"
        raise CaseSelectionError(msg) from e


def select_cases(
    df: pd.DataFrame, config: NnUNetDatasetConfig
) -> CaseSelection:
    """Resolve scans, masks, splits and case IDs for every configured collection.

    Parameters
    ----------
    df
        Query results, read with :func:`read_query_results`.
    config
        Validated dataset config.

    Raises
    ------
    CaseSelectionError
        If columns are missing, a selector is ambiguous, a configured ROI
        never occurs, a collection ends up with no cases, a split can't be
        made, there are no training cases, or two cases would get the same
        ID (ignoring letter case).
    """
    _check_columns(df, config)

    cases: list[SelectedCase] = []
    skipped: list[SkippedSample] = []
    for coll_cfg in config.collections:
        rows = df[df["Collection"] == coll_cfg.collection]
        coll_cases, coll_skipped = _resolve_collection(
            rows, coll_cfg, config.input_dir
        )
        if not coll_cases:
            _raise_no_cases(coll_cfg.collection, coll_skipped, config)
        splits = assign_splits(
            rows,
            [c.sample_id for c in coll_cases],
            coll_cfg,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        for case in coll_cases:
            case.split = splits[case.sample_id]
            case.case_id = make_case_id(
                case.collection,
                case.sample_id,
                prefix_with_collection=config.prefix_with_collection,
            )
        cases.extend(coll_cases)
        skipped.extend(coll_skipped)

    if not any(c.split == TRAIN for c in cases):
        msg = "No training cases selected; nnU-Net needs at least one."
        raise CaseSelectionError(msg)
    _check_unique_case_ids(cases)
    _warn_on_modality_mismatch(cases, config.channel_name)
    return CaseSelection(cases=cases, skipped=skipped)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_columns(df: pd.DataFrame, config: NnUNetDatasetConfig) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = (
            f"Query results are missing required columns {missing}. "
            "Is this a query_results.csv from 'imgnet query'?"
        )
        raise CaseSelectionError(msg)

    present = set(df["Collection"].dropna().unique())
    configured = [c.collection for c in config.collections]
    absent = [c for c in configured if c not in present]
    if absent:
        msg = (
            f"Collections {absent} are configured but have no rows in the "
            f"query results. Collections present: {sorted(present)}"
        )
        raise CaseSelectionError(msg)
    ignored = sorted(present - set(configured))
    if ignored:
        logger.warning(
            "Collections in the query results but not in the config are "
            "ignored: %s",
            ignored,
        )

    for coll_cfg in config.collections:
        unknown = sorted(coll_cfg.selector_columns() - set(df.columns))
        if unknown:
            msg = (
                f"{coll_cfg.collection}: columns {unknown} are not in the "
                f"query results. Available columns: {sorted(df.columns)}"
            )
            raise CaseSelectionError(msg)


def _check_unique_case_ids(cases: list[SelectedCase]) -> None:
    """Case IDs must be unique ignoring letter case.

    macOS and Windows file systems are case-insensitive by default, so
    ``MSWAL_M1`` and ``MSWAL_m1`` would overwrite each other's files.
    """
    by_id: dict[str, list[SelectedCase]] = {}
    for case in cases:
        by_id.setdefault(case.case_id.casefold(), []).append(case)
    clashes = [v for v in by_id.values() if len(v) > 1]
    if clashes:
        examples = [
            ", ".join(
                f"{c.collection}/{c.sample_id} -> {c.case_id}" for c in v
            )
            for v in clashes[:_MAX_EXAMPLES]
        ]
        msg = (
            f"{len(clashes)} case IDs are shared by more than one sample "
            "(IDs are compared ignoring letter case). Set "
            "prefix_with_collection to true, or check SampleIDs that differ "
            "only by letter case or removed characters:\n  "
            + "\n  ".join(examples)
        )
        raise CaseSelectionError(msg)


def _raise_no_cases(
    collection: str,
    skipped: list[SkippedSample],
    config: NnUNetDatasetConfig,
) -> None:
    reasons = Counter(s.reason.split(":", 1)[0] for s in skipped)
    msg = (
        f"{collection}: no cases selected; all {len(skipped)} samples were "
        f"skipped ({dict(reasons.most_common(_MAX_EXAMPLES))})."
    )
    not_found = [s for s in skipped if s.reason.startswith("file not found")]
    if not_found:
        msg += (
            " None of the selected files were found: check that input_dir "
            f"points at the download folder that contains {collection!r} "
            f"(usually 'srcdata'). input_dir is {config.input_dir}; first "
            f"missing file: {not_found[0].reason.split(': ', 1)[1]}"
        )
    raise CaseSelectionError(msg)


def _warn_on_modality_mismatch(
    cases: list[SelectedCase], channel_name: str
) -> None:
    """nnU-Net normalises by channel name, so modalities must agree."""
    by_modality = Counter(c.modality for c in cases if c.modality)
    if len(by_modality) > 1:
        logger.warning(
            "Selected scans have mixed modalities %s, but the dataset has "
            "one input channel (%r). nnU-Net would normalise them all the "
            "same way.",
            dict(by_modality),
            channel_name,
        )
    elif by_modality:
        (modality,) = by_modality
        if modality.casefold() != channel_name.casefold():
            logger.warning(
                "Selected scans are %s but channel_name is %r. nnU-Net "
                "chooses the normalisation from channel_name; set it to %r "
                "if that is what these scans are.",
                modality,
                channel_name,
                modality,
            )


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------


def select_rows(df: pd.DataFrame, selector: Selector) -> pd.Series:
    """Boolean mask of rows matching every entry of *selector*.

    - Strings and booleans are compared with the CSV text exactly.
    - Integers also match ``"1.0"``: ``imgnet query`` writes integer
      columns as floats when collections are combined.
    - Floats are compared numerically.
    - Missing values never match.
    """
    mask = pd.Series(True, index=df.index)
    for column, expected in selector.items():
        values = df[column]
        if isinstance(expected, Regex):
            matched = values.notna() & values.astype(str).str.contains(
                expected.regex, regex=True
            )
        else:
            allowed = expected if isinstance(expected, list) else [expected]
            matched = pd.Series(False, index=df.index)
            for value in allowed:
                matched |= _matches_value(values, value)
        mask &= matched
    return mask


def _matches_value(values: pd.Series, expected: object) -> pd.Series:
    # bool before int: bool is a subclass of int.
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return values == str(expected)
    if isinstance(expected, int):
        text = values.str.replace(_TRAILING_ZERO_DECIMALS, "", regex=True)
        return text == str(expected)
    return pd.to_numeric(values, errors="coerce") == expected


# ---------------------------------------------------------------------------
# Resolving scans and masks
# ---------------------------------------------------------------------------


@dataclass
class _Ambiguity:
    what: str
    examples: list[str] = field(default_factory=list)
    count: int = 0

    def add(self, sample_id: str, paths: list[str]) -> None:
        self.count += 1
        if len(self.examples) < _MAX_EXAMPLES:
            self.examples.append(f"{sample_id}: {paths}")


def _resolve_collection(
    rows: pd.DataFrame,
    cfg: CollectionConfig,
    input_dir: Path,
) -> tuple[list[SelectedCase], list[SkippedSample]]:
    name = cfg.collection
    base = select_rows(rows, cfg.selector)
    scans = rows[
        base
        & (rows["file_type"] == "scan")
        & select_rows(rows, cfg.scan_selector)
    ]
    masks = rows[
        base
        & (rows["file_type"] == "mask")
        & select_rows(rows, cfg.mask_selector)
    ]
    scans_by_sid = _paths_by(scans, ["SampleID"])
    modality_by_scan = _modality_by_filepath(scans)
    coll_dir = input_dir / name

    scan_ambiguity = _Ambiguity("scans")
    mask_ambiguity = _Ambiguity("masks")
    cases: list[SelectedCase] = []
    skipped: list[SkippedSample] = []

    if cfg.method == "combine":
        _check_rois_exist(masks, cfg)
        masks = masks[masks[cfg.roi_column].isin(list(cfg.intensity_map))]
        masks_by_key = _paths_by(masks, ["SampleID", cfg.roi_column])
    else:
        masks_by_key = _paths_by(masks, ["SampleID"])

    for sid in _unique_sorted(rows["SampleID"]):
        scan_paths = scans_by_sid.get((sid,), [])
        if not scan_paths:
            skipped.append(SkippedSample(name, sid, "no scan matched"))
            continue
        if len(scan_paths) > 1:
            scan_ambiguity.add(sid, scan_paths)
            continue
        case = SelectedCase(
            name,
            sid,
            coll_dir / scan_paths[0],
            cfg.method,
            [],
            modality=modality_by_scan.get(scan_paths[0]),
        )
        attach = _attach_roi_masks if cfg.method == "combine" else _attach_mask
        reason = attach(case, coll_dir, cfg, masks_by_key, mask_ambiguity)
        if reason is None:
            cases.append(case)
        elif reason != _AMBIGUOUS:
            skipped.append(SkippedSample(name, sid, reason))

    for ambiguity in (scan_ambiguity, mask_ambiguity):
        if ambiguity.count:
            msg = (
                f"{name}: {ambiguity.count} samples match more than one "
                f"{ambiguity.what[:-1]} where one is expected. Add a "
                "selector to pick one, e.g. on a phase or filepath column. "
                "Examples:\n  " + "\n  ".join(ambiguity.examples)
            )
            raise CaseSelectionError(msg)

    cases, missing = _drop_missing_files(cases)
    skipped.extend(missing)
    return cases, skipped


def _check_rois_exist(masks: pd.DataFrame, cfg: CollectionConfig) -> None:
    """A configured ROI with no mask rows at all is almost always a typo."""
    available = set(masks[cfg.roi_column].dropna().unique())
    unknown = [roi for roi in cfg.intensity_map if roi not in available]
    if not unknown:
        return
    hints = {
        roi: difflib.get_close_matches(roi, available, n=3) for roi in unknown
    }
    shown = sorted(available)[:30]
    more = f" (+{len(available) - 30} more)" if len(available) > 30 else ""
    msg = (
        f"{cfg.collection}: ROIs {unknown} have no masks in the query "
        f"results (after the selectors). Close matches: {hints}. "
        f"Available {cfg.roi_column!r} values: {shown}{more}"
    )
    raise CaseSelectionError(msg)


def _modality_by_filepath(scans: pd.DataFrame) -> dict[str, str]:
    if "Modality" not in scans.columns:
        return {}
    known = scans.dropna(subset=["filepath", "Modality"])
    return dict(zip(known["filepath"], known["Modality"], strict=True))


_AMBIGUOUS = "<ambiguous>"
"""Returned by the attach functions when the sample was recorded as ambiguous."""


def _attach_mask(
    case: SelectedCase,
    coll_dir: Path,
    cfg: CollectionConfig,
    masks_by_key: dict[tuple[str, ...], list[str]],
    ambiguity: _Ambiguity,
) -> str | None:
    """remap: set the case's single mask; return a skip reason or None."""
    paths = masks_by_key.get((case.sample_id,), [])
    if not paths:
        return "no mask matched"
    if len(paths) > 1:
        ambiguity.add(case.sample_id, paths)
        return _AMBIGUOUS
    case.masks = [coll_dir / paths[0]]
    case.remap = {int(k): v for k, v in cfg.intensity_map.items()}
    return None


def _attach_roi_masks(
    case: SelectedCase,
    coll_dir: Path,
    cfg: CollectionConfig,
    masks_by_key: dict[tuple[str, ...], list[str]],
    ambiguity: _Ambiguity,
) -> str | None:
    """combine: set one mask per ROI; return a skip reason or None."""
    missing: list[str] = []
    for roi, label in cfg.intensity_map.items():
        paths = masks_by_key.get((case.sample_id, roi), [])
        if len(paths) > 1:
            ambiguity.add(f"{case.sample_id} ({roi})", paths)
            return _AMBIGUOUS
        if not paths:
            missing.append(roi)
            continue
        case.masks.append(coll_dir / paths[0])
        case.labels.append(label)
    if not case.masks:
        return "no configured ROI matched"
    if missing and not cfg.allow_missing_rois:
        return f"missing ROIs {missing}"
    return None


def _drop_missing_files(
    cases: list[SelectedCase],
) -> tuple[list[SelectedCase], list[SkippedSample]]:
    kept: list[SelectedCase] = []
    skipped: list[SkippedSample] = []
    for case in cases:
        files = [case.scan, *case.masks]
        absent = [str(p) for p in files if not p.is_file()]
        if absent:
            reason = f"file not found: {absent[0]}"
            if len(absent) > 1:
                reason += f" (+{len(absent) - 1} more)"
            skipped.append(
                SkippedSample(case.collection, case.sample_id, reason)
            )
        else:
            kept.append(case)
    return kept, skipped


def _paths_by(
    rows: pd.DataFrame, keys: list[str]
) -> dict[tuple[str, ...], list[str]]:
    out: dict[tuple[str, ...], list[str]] = {}
    clean = rows.dropna(subset=[*keys, "filepath"])
    for key_values, path in zip(
        clean[keys].itertuples(index=False, name=None),
        clean["filepath"],
        strict=True,
    ):
        paths = out.setdefault(tuple(key_values), [])
        if path not in paths:
            paths.append(path)
    return out


def _unique_sorted(values: pd.Series) -> list[str]:
    return sorted(values.dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def assign_splits(
    rows: pd.DataFrame,
    sample_ids: list[str],
    cfg: CollectionConfig,
    *,
    test_size: float,
    random_state: int,
) -> dict[str, str]:
    """Map each sample ID to ``"Tr"`` or ``"Ts"``.

    Uses ``cfg.split`` when that column has values for this collection,
    otherwise a seeded random split grouped by ``cfg.group_column``.
    """
    if not sample_ids:
        return {}
    all_columns = list(rows.columns)
    rows = rows[rows["SampleID"].isin(sample_ids)]
    if cfg.split is not None:
        column = cfg.split.column
        if column in rows.columns and rows[column].notna().any():
            if cfg.group_column is not None:
                logger.warning(
                    "%s: using split column %r; group_column %r only "
                    "applies to random splits and is ignored.",
                    cfg.collection,
                    column,
                    cfg.group_column,
                )
            return _splits_from_column(rows, sample_ids, cfg)
        if column in all_columns:
            problem = "has no values for this collection"
        else:
            close = difflib.get_close_matches(column, all_columns, n=3)
            problem = "is not in the query results" + (
                f" (did you mean {close}?)" if close else ""
            )
        logger.warning(
            "%s: split column %r %s; using a random split instead%s.",
            cfg.collection,
            column,
            problem,
            f", grouped by {cfg.group_column!r}" if cfg.group_column else "",
        )
    return _random_splits(rows, sample_ids, cfg, test_size, random_state)


def _one_value_per_sample(
    rows: pd.DataFrame,
    sample_ids: list[str],
    column: str,
    collection: str,
) -> dict[str, str | None]:
    per_sample = rows.dropna(subset=[column]).groupby("SampleID")[column]
    values = per_sample.unique().to_dict()
    conflicting = {sid: list(v) for sid, v in values.items() if len(v) > 1}
    if conflicting:
        examples = list(conflicting.items())[:_MAX_EXAMPLES]
        msg = (
            f"{collection}: {len(conflicting)} samples have more than one "
            f"{column!r} value, e.g. {examples}"
        )
        raise CaseSelectionError(msg)
    return {
        sid: (values[sid][0] if sid in values else None) for sid in sample_ids
    }


def _splits_from_column(
    rows: pd.DataFrame, sample_ids: list[str], cfg: CollectionConfig
) -> dict[str, str]:
    assert cfg.split is not None
    column = cfg.split.column
    values = _one_value_per_sample(rows, sample_ids, column, cfg.collection)
    train, test = set(cfg.split.train), set(cfg.split.test)
    out: dict[str, str] = {}
    unmapped: set[str] = set()
    no_value: list[str] = []
    for sid, value in values.items():
        if value is None:
            no_value.append(sid)
        elif value in train:
            out[sid] = TRAIN
        elif value in test:
            out[sid] = TEST
        else:
            unmapped.add(value)
    if no_value:
        msg = (
            f"{cfg.collection}: {len(no_value)} samples have no {column!r} "
            f"value, e.g. {no_value[:_MAX_EXAMPLES]}"
        )
        raise CaseSelectionError(msg)
    if unmapped:
        msg = (
            f"{cfg.collection}: {column!r} values {sorted(unmapped)} are "
            "not listed in split.train or split.test"
        )
        raise CaseSelectionError(msg)
    return out


def _random_splits(
    rows: pd.DataFrame,
    sample_ids: list[str],
    cfg: CollectionConfig,
    test_size: float,
    random_state: int,
) -> dict[str, str]:
    if cfg.group_column is None:
        groups = {sid: sid for sid in sample_ids}
    else:
        values = _one_value_per_sample(
            rows, sample_ids, cfg.group_column, cfg.collection
        )
        no_group = [sid for sid, v in values.items() if v is None]
        if no_group:
            logger.warning(
                "%s: %d samples have no %r value; each is split as its own "
                "group.",
                cfg.collection,
                len(no_group),
                cfg.group_column,
            )
        # Prefix so a SampleID can never equal a real group value.
        groups = {
            sid: (f"group:{v}" if v is not None else f"sample:{sid}")
            for sid, v in values.items()
        }

    unique_groups = set(groups.values())
    n_groups = len(unique_groups)
    n_test = 0
    if test_size > 0:
        if n_groups < 2:
            logger.warning(
                "%s: only %d split group(s); all samples go to training.",
                cfg.collection,
                n_groups,
            )
        else:
            n_test = min(
                math.ceil(test_size * n_groups - _SPLIT_EPSILON),
                n_groups - 1,
            )

    # Each group gets a fixed pseudo-random rank from (seed, collection,
    # group), and the lowest ranks form the test set. A group's rank
    # doesn't depend on which other groups exist, so adding or removing a
    # sample moves at most one other group across the split, and other
    # collections never affect this one.
    ranked = sorted(
        unique_groups,
        key=lambda g: _split_rank(random_state, cfg.collection, g),
    )
    test_groups = set(ranked[:n_test])
    return {
        sid: (TEST if group in test_groups else TRAIN)
        for sid, group in groups.items()
    }


def _split_rank(random_state: int, collection: str, group: str) -> bytes:
    key = f"{random_state}\0{collection}\0{group}".encode()
    return hashlib.blake2b(key, digest_size=16).digest()


# ---------------------------------------------------------------------------
# Case IDs
# ---------------------------------------------------------------------------


def make_case_id(
    collection: str, sample_id: str, *, prefix_with_collection: bool
) -> str:
    """Build a file-name-safe case ID.

    Characters other than letters, digits, ``_`` and ``-`` are removed, e.g.
    ``("Abdomen Atlas 3", "123")`` -> ``"AbdomenAtlas3_123"``.
    """
    sid = _INVALID_ID_CHARS.sub("", sample_id)
    if not sid:
        msg = f"{collection}: SampleID {sample_id!r} has no usable characters"
        raise CaseSelectionError(msg)
    if not prefix_with_collection:
        return sid
    prefix = _INVALID_ID_CHARS.sub("", collection)
    if not prefix:
        msg = f"Collection name {collection!r} has no usable characters"
        raise CaseSelectionError(msg)
    return f"{prefix}_{sid}"
