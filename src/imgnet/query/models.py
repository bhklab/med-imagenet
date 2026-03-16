import base64
import operator
import re
import zlib
from typing import Any

import msgpack
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

    tag: str = Field(
        description="The DICOM tag the rule applies to.",
    )
    value: str | list[str] = Field(
        description="The value to compare against.",
    )
    comparison: str = Field(
        description=(
            "The comparison type. `==`/`!=` interpret the comparison value as regex."
        ),
    )

    def evaluate(self, dicom_element: dict) -> bool:  # noqa: PLR0911 PLR0912
        """Evaluate whether a DICOM metadata dict is accepted by this rule."""
        tag_value = dicom_element.get(self.tag)
        if tag_value is None:
            return False

        if isinstance(tag_value, str):
            if tag_value.strip().startswith(
                "["
            ) and tag_value.strip().endswith("]"):
                matches = re.findall(
                    r"""(['"])(.*?)\1|([^'",\s\[\]]+)""", tag_value
                )
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
                    msg = f"{self.comparison} comparison only compatible with numeric arguments, not list."
                    raise InvalidComparisonError(msg)
                for element in tag_value:
                    if element == "" or element is None:
                        return False
                    try:
                        if not op_fn(float(element), float(self.value)):
                            return False
                    except ValueError as exc:
                        msg = (
                            f"'{self.comparison}' comparisons only support numeric values."
                            f"\nInput: {self.tag}: {tag_value}, {self.comparison} {self.value}"
                        )
                        raise RuleError(msg) from exc
                return True

        return False

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series over *df*: True for rows accepted by this rule.

        Vectorized counterpart of ``evaluate`` for bulk filtering.
        """
        col = df.get(self.tag)
        if col is None:
            return pd.Series(False, index=df.index)

        col = col.astype(str)

        match self.comparison:
            case "==" | "=":
                patterns = (
                    [self.value] if isinstance(self.value, str) else self.value
                )
                combined = "|".join(f"(?:{p})" for p in patterns)
                return col.str.match(combined, na=False)

            case "!=":
                patterns = (
                    [self.value] if isinstance(self.value, str) else self.value
                )
                combined = "|".join(f"(?:{p})" for p in patterns)
                return ~col.str.match(combined, na=False)

            case ">" | "<" | ">=" | "<=":
                if isinstance(self.value, list):
                    msg = f"{self.comparison} comparison only compatible with numeric arguments, not list."
                    raise InvalidComparisonError(msg)
                op_fn = NUMERIC_OPS[self.comparison]
                return op_fn(
                    pd.to_numeric(col, errors="coerce"), float(self.value)
                ).fillna(False)

        return pd.Series(False, index=df.index)


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

    @field_validator("rules", mode="before")
    def validate_rules(
        self, value: Any
    ) -> dict[str, Rule | list[Rule]] | None:  # noqa: PLR0912 ANN401
        if value is None:
            return None
        if not isinstance(value, dict):
            msg = f"rules must be a dict, got {type(value)}."
            raise RulesValidationError(msg)

        from imgnet.query.parser import parse_rule

        result: dict[str, Rule | list[Rule]] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"rules keys must be str, got {type(key)}."
                raise RulesValidationError(msg)
            if isinstance(item, dict):
                result[key] = Rule.model_validate(item)
            elif isinstance(item, str):
                result[key] = parse_rule(item)
            elif isinstance(item, Rule):
                result[key] = item
            elif isinstance(item, list):
                if all(isinstance(entry, dict) for entry in item):
                    result[key] = [
                        Rule.model_validate(entry) for entry in item
                    ]
                elif all(isinstance(entry, str) for entry in item):
                    result[key] = [parse_rule(entry) for entry in item]
                elif all(isinstance(entry, Rule) for entry in item):
                    result[key] = item
                else:
                    msg = f"rules[{key!r}] list elements must all be str, dict, or Rule."
                    raise RulesValidationError(msg)
            else:
                msg = f"rules[{key!r}] must be str, dict, Rule, or list - got {type(item)}."
                raise RulesValidationError(msg)
        return result

    def to_token(self) -> str:
        raw = self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude_unset=True,
        )
        packed = msgpack.packb(raw)
        compressed = zlib.compress(packed, level=9)
        return base64.urlsafe_b64encode(compressed).decode()

    @classmethod
    def from_token(cls, token: str) -> "ValidQuery":
        compressed = base64.urlsafe_b64decode(token)
        packed = zlib.decompress(compressed)
        data = msgpack.unpackb(packed)
        return cls.model_validate(data)

    def process(self, store: IndexedDatasets) -> pd.DataFrame:
        """Return a DataFrame containing selected SeriesInstanceUID rows."""
        from imgnet.query.engine import run_query

        query_results = run_query(self, store)
        logger.info(f"Found {len(query_results)} matches to the query.")
        return query_results
