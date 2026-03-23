from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pandas as pd
from imgtools.dicom import Interlacer

from imgnet.collections.source import FileType
from imgnet.loggers import logger
from imgnet.query.models import CollectionsValidationError

if TYPE_CHECKING:
    from imgnet.collections.store import IndexedDatasets
    from imgnet.query.models import Rule, ValidQuery


def run_query(
    valid_query: "ValidQuery", store: IndexedDatasets
) -> pd.DataFrame:
    """Execute a `ValidQuery` against the indexed datasets store."""

    supported = store.collections

    requested_file_type = valid_query.file_type

    collections = valid_query.collections
    modality_queries = valid_query.modalities
    rules = valid_query.rules

    if collections == "all":
        if requested_file_type in (None, "all"):
            collections = supported
        else:
            collections = [
                c
                for c in supported
                if store.file_type(c) == requested_file_type
            ]
    if isinstance(collections, str):
        collections = [collections]
    for collection in collections:
        if collection not in supported:
            msg = f"Collection {collection} not found."
            raise CollectionsValidationError(msg)
        if requested_file_type not in (None, "all"):
            expected = (
                requested_file_type.value
                if isinstance(requested_file_type, FileType)
                else requested_file_type
            )
            actual = store.file_type(collection).value
            if store.file_type(collection) != requested_file_type:
                msg = (
                    f"Collection {collection!r} is of type {actual}, "
                    f"but query requested {expected}."
                )
                raise CollectionsValidationError(msg)

    if isinstance(modality_queries, str):
        modality_queries = [modality_queries]

    def _query_one(collection: str) -> pd.DataFrame:
        file_type = store.file_type(collection)
        if file_type == FileType.DICOM:
            return _run_query_dicom(collection, store, modality_queries, rules)
        elif file_type == FileType.NIFTI:
            return _run_query_nifti(collection, store, modality_queries, rules)
        else:
            msg = f"Unsupported file type for collection {collection}: {file_type}"
            raise ValueError(msg)

    logger.info("Running query...")

    if len(collections) == 1:
        matches = [_query_one(collections[0])]
    else:
        matches = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_query_one, col): col for col in collections
            }
            for future in as_completed(futures):
                matches.append(future.result())

    return pd.concat(matches, ignore_index=True)


def _run_query_dicom(
    collection: str,
    store: IndexedDatasets,
    modality_queries: list[str],
    rules: dict[str, Rule | list[Rule]] | None,
) -> pd.DataFrame:
    from imgnet.query.models import Rule

    index_df = store.index(collection)
    crawl_db = store.crawl_db(collection)

    csv_path = store.imgtools_path / collection / "index.csv"
    interlacer = Interlacer(csv_path)
    modality_matches = []

    for query in modality_queries:
        if query == "all" or query is None:
            query_result = interlacer.query_all()
        else:
            query_result = interlacer.query(query)
        modality_matches += [
            [node.SeriesInstanceUID for node in group]
            for group in query_result
        ]

    if not modality_matches:
        index_df = index_df.iloc[0:0].copy()
        index_df["Collection"] = collection
        return index_df

    # Flatten crawl_db metadata into a DataFrame keyed by SeriesInstanceUID
    all_uids = [uid for group in modality_matches for uid in group]
    records = {}
    for uid in all_uids:
        entry = crawl_db[uid]
        records[uid] = entry[next(iter(entry))]
    meta_df = pd.DataFrame.from_dict(records, orient="index")

    # Vectorized rule evaluation via Rule.mask()
    accepted = pd.Series(True, index=meta_df.index)
    if rules and "Modality" in meta_df.columns:
        for modality, modality_rules in rules.items():
            rules_list = (
                [modality_rules]
                if isinstance(modality_rules, Rule)
                else modality_rules
            )
            is_mod = meta_df["Modality"] == modality
            rule_mask = is_mod.copy()
            for rule in rules_list:
                rule_mask &= rule.mask(meta_df)
            accepted &= ~is_mod | rule_mask

    accepted_uids = set(meta_df.index[accepted])

    # If root node rejected, skip entire group
    result: list[str] = []
    for group in modality_matches:
        if group[0] not in accepted_uids:
            continue
        result.extend(uid for uid in group if uid in accepted_uids)

    index_df = index_df[index_df["SeriesInstanceUID"].isin(result)].copy()
    index_df["Collection"] = collection
    return index_df


def _run_query_nifti(
    collection: str,
    store: IndexedDatasets,
    modality_queries: list[str],
    rules: dict[str, Rule | list[Rule]] | None,
) -> pd.DataFrame:
    from imgnet.query.models import Rule

    index_df = store.index(collection)

    # Vectorized modality filtering
    modality_mask = pd.Series(False, index=index_df.index)
    for query in modality_queries:
        if query == "all" or query is None:
            modality_mask[:] = True
            break
        elif "," in query:
            modality_mask |= index_df["Modality"].isin(query.split(","))
        else:
            modality_mask |= index_df["Modality"] == query

    filtered = index_df[modality_mask]

    # Vectorized rule evaluation via Rule.mask()
    if rules:
        for modality, modality_rules in rules.items():
            rules_list = (
                [modality_rules]
                if isinstance(modality_rules, Rule)
                else modality_rules
            )
            is_mod = filtered["Modality"] == modality
            rule_mask = is_mod.copy()
            for rule in rules_list:
                rule_mask &= rule.mask(filtered)
            filtered = filtered[~is_mod | rule_mask]

    filtered = filtered.copy()
    filtered["Collection"] = collection
    return filtered
