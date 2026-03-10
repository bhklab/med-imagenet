import pandas as pd
from imgtools.dicom import Interlacer

from imgnet.collections.store import IndexedDatasets
from imgnet.query.models import CollectionsValidationError


def run_query(valid_query, store: IndexedDatasets) -> pd.DataFrame:
    """Execute a `ValidQuery` against the indexed datasets store."""
    from imgnet.query.models import Rule

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
    for collection in collections:
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
        matches.append(index_df)

    return pd.concat(matches, ignore_index=True)
