import operator
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from imgnet.collections.store import IndexedDatasets
from imgnet.loggers import logger


NUMERIC_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


class RuleError(Exception):
    """Exception raised for invalid rules."""


class InvalidComparisonError(RuleError):
    """Exception raised when a Rule has an invalid comparison type."""


class ValidQueryError(Exception):
    """Base exception for ValidQuery errors."""


class ModalitiesValidationError(ValidQueryError):
    """Raised when modality field validation fails."""


class CollectionsValidationError(ValidQueryError):
    """Raised when collections field validation fails."""


class RulesValidationError(ValidQueryError):
    """Raised when rules field validation fails."""


class RulesValidationParsingError(RulesValidationError):
    """Raised when parsing a Rule from string fails."""


class Rule(BaseModel):
    """Comparison rule between one DICOM tag and a value or list of values."""

    tag: str = Field(description="The DICOM tag the rule applies to.")
    value: str | list[str] = Field(description="The value to compare against.")
    comparison: str = Field(
        description=(
            "The comparison type. `==`/`!=` interpret the comparison value as regex."
        )
    )

    def evaluate(self, dicom_element: dict) -> bool:
        """Evaluate whether a DICOM metadata dict is accepted by this rule."""
        tag_value = dicom_element.get(self.tag)
        if tag_value is None:
            return False

        if isinstance(tag_value, str):
            if tag_value.strip().startswith("[") and tag_value.strip().endswith("]"):
                matches = re.findall(r"""(['"])(.*?)\1|([^'",\s\[\]]+)""", tag_value)
                tag_value = [m[1] if m[1] else m[2] for m in matches]
            else:
                tag_value = [tag_value.strip()]

        match self.comparison:
            case "==" | "=":
                patterns = self.value
                if isinstance(self.value, str):
                    patterns = [self.value]
                for element in tag_value:
                    for pattern in patterns:
                        if re.match(pattern, element):
                            return True
                return False

            case "!=":
                patterns = self.value
                if isinstance(self.value, str):
                    patterns = [self.value]
                for element in tag_value:
                    for pattern in patterns:
                        if re.match(pattern, element):
                            return False
                return True

            case ">" | "<" | ">=" | "<=":
                op_fn = NUMERIC_OPS[self.comparison]
                if isinstance(self.value, list):
                    raise InvalidComparisonError(
                        f"{self.comparison} comparison only compatible with numeric arguments, not list."
                    )
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        if not op_fn(float(element), float(self.value)):
                            return False
                    except ValueError as exc:
                        raise RuleError(
                            f"'{self.comparison}' comparisons only support numeric values."
                            f"\nInput: {self.tag}: {tag_value}, {self.comparison} {self.value}"
                        ) from exc
                return True

        return False


class ValidQuery(BaseModel):
    """Pydantic model representing a Med-ImageNet query."""

    collections: str | list[str] = Field(
        description="The collections to query",
        default="all",
        examples=["all", "4D-Lung", ["4D-Lung", "RADCURE"]],
    )
    modalities: str | list[str] = Field(
        description="The modalities to query",
        default="all",
        examples=["all", "MR,RTSTRUCT", ["MR,RTSTRUCT", "CT,RTSTRUCT"]],
    )
    rules: dict[str, Rule | list[Rule]] | None = Field(
        description="The query filter rules, optionally grouped by modality",
        default=None,
        examples=[
            {
                "RTSTRUCT": "ROINames == ['lung', 'lung.*']",
                "MR": ["ImageType=='PRIMARY'", "PixelPaddingValue == 1"],
            }
        ],
    )

    @field_validator("collections", mode="after")
    def validate_collections(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list):
            if not all(isinstance(collection, str) for collection in value):
                raise CollectionsValidationError(
                    "All collection names must be strings."
                )
        elif not isinstance(value, str):
            raise CollectionsValidationError(
                f"collections must be str | list[str], got {type(value)}."
            )
        return value

    @field_validator("modalities", mode="before")
    def validate_modalities(cls, value: Any) -> str | list[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ModalitiesValidationError(
            f"modalities must be of type str | list[str]. Got type {type(value)} instead."
        )

    @field_validator("rules", mode="before")
    def validate_rules(cls, value: Any) -> dict[str, Rule | list[Rule]] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RulesValidationError(f"rules must be a dict, got {type(value)}.")

        from imgnet.query.parser import parse_rule

        result: dict[str, Rule | list[Rule]] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RulesValidationError(
                    f"rules keys must be str, got {type(key)}."
                )
            if isinstance(item, dict):
                result[key] = Rule.model_validate(item)
            elif isinstance(item, str):
                result[key] = parse_rule(item)
            elif isinstance(item, Rule):
                result[key] = item
            elif isinstance(item, list):
                if all(isinstance(entry, dict) for entry in item):
                    result[key] = [Rule.model_validate(entry) for entry in item]
                elif all(isinstance(entry, str) for entry in item):
                    result[key] = [parse_rule(entry) for entry in item]
                elif all(isinstance(entry, Rule) for entry in item):
                    result[key] = item
                else:
                    raise RulesValidationError(
                        f"rules[{key!r}] list elements must all be str, dict, or Rule."
                    )
            else:
                raise RulesValidationError(
                    f"rules[{key!r}] must be str, dict, Rule, or list - got {type(item)}."
                )
        return result

    def process(self, store: IndexedDatasets) -> pd.DataFrame:
        """Return a DataFrame containing selected SeriesInstanceUID rows."""
        from imgnet.query.engine import run_query

        query_results = run_query(self, store)
        logger.info(f"Found {len(query_results)} matches to the query.")
        return query_results
