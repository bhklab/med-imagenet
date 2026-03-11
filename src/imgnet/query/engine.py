from __future__ import annotations

import pandas as pd
from imgtools.dicom import Interlacer
from typing import TYPE_CHECKING

from imgnet.collections.store import IndexedDatasets
from imgnet.collections.source import FileType
from imgnet.query.models import CollectionsValidationError
from imgnet.loggers import logger

if TYPE_CHECKING:
    from imgnet.query.models import ValidQuery, Rule


def run_query(valid_query: "ValidQuery", store: IndexedDatasets) -> pd.DataFrame:
    """Execute a `ValidQuery` against the indexed datasets store."""

    supported = store.collections

    collections = valid_query.collections
    modality_queries = valid_query.modalities
    rules = valid_query.rules

    if collections == "all":
        collections = supported
    if isinstance(collections, str):
        collections = [collections]
    for collection in collections:
        if collection not in supported:
            raise CollectionsValidationError(f"Collection {collection} not found.")

    if isinstance(modality_queries, str):
        modality_queries = [modality_queries]

    matches = []
    logger.info("Running query...")
    for collection in collections:        
        if store.file_type(collection) == FileType.DICOM:
            matches.append(_run_query_dicom(collection, store, modality_queries, rules))
        elif store.file_type(collection) == FileType.NIFTI:
            matches.append(_run_query_nifti(collection, store, modality_queries, rules))
        else:
            raise ValueError(f"Unsupported file type for collection {collection}: {store.file_type(collection)}")

    return pd.concat(matches, ignore_index=True)

def _run_query_dicom(
    collection: str,
    store: IndexedDatasets,
    modality_queries: list[str],
    rules: dict[str, Rule | list[Rule]],
):
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
            [node.SeriesInstanceUID for node in group] for group in query_result
        ]

    result = []
    for group in modality_matches:
        for series in group:
            # crawl_db nests an extra key layer between series UID and metadata
            for key in crawl_db[series]:
                dicom = crawl_db[series][key]

            modality = dicom["Modality"]
            accept_series = True
            if rules:
                modality_rules = rules.get(modality)
                if isinstance(modality_rules, Rule):
                    modality_rules = [modality_rules]
                if modality_rules:
                    for rule in modality_rules:
                        if not rule.evaluate(dicom):
                            accept_series = False
                            break

            if accept_series:
                result.append(series)
            elif series == group[0]:
                # If the root node is not selected, children are skipped to avoid
                # selecting masks that reference an unselected DICOM.
                break

    index_df = index_df[index_df["SeriesInstanceUID"].isin(result)]
    index_df["Collection"] = collection
    return index_df

def _run_query_nifti(
    collection: str,
    store: IndexedDatasets,
    modality_queries: list[str],
    rules: dict[str, Rule | list[Rule]],
):
    from imgnet.query.models import Rule

    index_df = store.index(collection)

    modality_matches = []

    for query in modality_queries:
        if query == "all" or query is None:
            query_result = [[idx] for idx in index_df.index.tolist()]
        else:
            if "," in query:
                query_result = [
                    group.index.tolist() 
                    for _, group in index_df[index_df["Modality"].isin(query.split(","))].groupby("reference_id")
                ]
            else:
                query_result = [[idx] for idx in index_df[index_df["Modality"] == query].index.tolist()]
            
        modality_matches += query_result

    result = []
    for group in modality_matches:
        for series_index in group:

            series = index_df.iloc[series_index]

            modality = series["Modality"]
            accept_series = True
            if rules:
                modality_rules = rules.get(modality)
                if isinstance(modality_rules, Rule):
                    modality_rules = [modality_rules]
                if modality_rules:
                    for rule in modality_rules:
                        if not rule.evaluate(series):
                            accept_series = False
                            break

            if accept_series:
                result.append(series_index)

    index_df = index_df.iloc[result]
    index_df["Collection"] = collection
    return index_df