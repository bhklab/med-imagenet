from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Annotated, Union

import pandas as pd
from pydantic import BaseModel, Field

from imgnet.collections.source import FileType
from imgnet.loggers import logger
from imgnet.query.rule import BoolRule, ConditionRule

if TYPE_CHECKING:
    from imgnet.collections.store import IndexedDatasets

Rule = Annotated[Union[ConditionRule, BoolRule], Field(discriminator="kind")]


class ValidQuery(BaseModel):
    """Pydantic model representing a Med-ImageNet query."""

    collections: list[str] | None = Field(
        description="The collections to query",
        default=None,
        examples=["all", "4D-Lung", ["4D-Lung", "RADCURE"]],
    )
    modalities: list[str] | None = Field(
        description="The modalities to query",
        default=None,
        examples=["MR", "RTSTRUCT", ["MR", "RTSTRUCT"]],
    )
    rules: list[Rule] | None = Field(
        description="The query filter rules, optionally grouped by modality",
        default=None,
        examples=[
            "MR(ImageType=='PRIMARY' AND PixelPaddingValue == 1)",
            "RTSTRUCT(ROINames == ['lung', 'lung.*'])",
        ],
    )
    file_type: FileType | None = Field(
        description="The file type to query",
        default=None,
        examples=[FileType.DICOM.value, FileType.NIFTI.value],
    )

    def process(self, store: IndexedDatasets) -> pd.DataFrame:
        supported_collections = store.collections
        requested_file_type = self.file_type
        requested_collections = self.collections
        requested_modalities = self.modalities
        rules = self.rules

        if requested_collections is None:
            if requested_file_type is None:
                collections = supported_collections
            else:
                collections = [
                    c
                    for c in supported_collections
                    if store.file_type(c) == requested_file_type
                ]
        else:
            collections = list(requested_collections)
            if collections == ["all"] or collections == ["ALL"]:
                collections = list(supported_collections)

        logger.debug("Resolved collections: %s", collections)

        for collection_name in collections:
            collection_obj = store.get_collection(collection_name)
            if requested_file_type is not None:
                expected = (
                    requested_file_type.value
                    if isinstance(requested_file_type, FileType)
                    else requested_file_type
                )
                actual_ft = collection_obj.file_type
                if actual_ft != requested_file_type:
                    msg = (
                        f"Collection {collection_name!r} is of type {actual_ft.value}, "
                        f"but query requested {expected}."
                    )
                    raise ValueError(msg)

        logger.info("Running query...")

        rules_list = list(rules) if rules is not None else []

        if len(collections) == 1:
            matches = [
                _run_query(
                    collections[0],
                    store,
                    requested_modalities,
                    rules_list,
                )
            ]
        else:
            matches = []
            with ThreadPoolExecutor() as executor:
                future_by_col = {
                    col: executor.submit(
                        _run_query,
                        col,
                        store,
                        requested_modalities,
                        rules_list,
                    )
                    for col in collections
                }
                for col in collections:
                    matches.append(future_by_col[col].result())

        if not matches:
            return pd.DataFrame()

        return pd.concat(matches, ignore_index=True)

    def __repr__(self) -> str:
        return (
            f"ValidQuery(collections={self.collections}, modalities={self.modalities}, "
            f"rules={self.rules}, file_type={self.file_type})"
        )


def _filter_modalities(
    modalities: list[str] | None, index: pd.DataFrame
) -> pd.DataFrame:
    if modalities is None:
        return index
    return index[index["Modality"].isin(modalities)].copy()


def _run_query(
    collection: str,
    store: IndexedDatasets,
    modalities: list[str] | None,
    rules: list[Rule],
) -> pd.DataFrame:
    try:
        col = store.get_collection(collection)

        _filtered = col.read_index_rows(modalities)
        sample_ids = set(_filtered["SampleID"].unique())
        if len(sample_ids) == 0:
            logger.warning(
                "No modalities matches found for collection %s.", collection
            )
            return pd.DataFrame()

        for rule in rules:
            rule_modality = rule.modality
            if modalities is not None and rule_modality not in modalities:
                logger.warning(
                    "Rule %s is for modality %s, but requested modalities are %s.",
                    rule,
                    rule_modality,
                    modalities,
                )
                continue
            _filtered = col.read_index_rows([rule_modality])
            mask = _filtered.apply(rule.evaluate, axis=1)

            if len(_filtered[mask]) == 0:
                logger.warning(
                    "No rules matches found for collection %s and rule %s.",
                    collection,
                    rule,
                )
                sample_ids = set()
                break
            chosen_sample_ids = set(_filtered[mask]["SampleID"].unique())
            sample_ids = sample_ids.intersection(chosen_sample_ids)

        # All modalities for surviving SampleIDs; Parquet filter avoids a full pandas index.
        result = col.read_index_rows(
            modalities=None,
            sample_ids=sample_ids,
        ).copy()
        result["Collection"] = collection
        return result
    except Exception as e:
        logger.error(
            "Error running query for collection %s: %s", collection, e
        )
        return pd.DataFrame()

