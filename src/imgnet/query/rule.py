from __future__ import annotations

import operator
import re
from typing import Literal

import pandas as pd
from lark import Lark, Transformer
from pydantic import BaseModel, Field

GRAMMAR = r"""
start: modality "(" expr ")"
modality: WORD

?expr: expr "OR" term   -> or_expr
     | term

?term: term "AND" factor -> and_expr
     | factor

?factor: condition
       | "(" expr ")"

condition: WORD OP value

value: ESCAPED_STRING | NUMBER | WORD

OP: "==" | "!=" | "<=" | ">=" | "<" | ">"

%import common.WORD
%import common.NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


PARSER = Lark(GRAMMAR, start="start")


def _format_rule_repr(node: ConditionRule | BoolRule, indent: int = 0) -> str:
    """Multiline repr with 2-space indents; nested BoolRule children step in further."""
    pad = " " * indent
    step = 2
    inner = " " * (indent + step)
    child_indent = indent + 2 * step

    if isinstance(node, ConditionRule):
        return (
            f"{pad}ConditionRule(\n"
            f"{inner}modality={repr(node.modality)},\n"
            f"{inner}tag={repr(node.tag)},\n"
            f"{inner}comparison={repr(node.comparison)},\n"
            f"{inner}value={repr(node.value)}\n"
            f"{pad})"
        )

    left = _format_rule_repr(node.left, child_indent)
    right = _format_rule_repr(node.right, child_indent)
    return (
        f"{pad}BoolRule(\n"
        f"{inner}modality={repr(node.modality)},\n"
        f"{inner}op={repr(node.op)},\n"
        f"{inner}left=\n{left},\n"
        f"{inner}right=\n{right}\n"
        f"{pad})"
    )


NUMERIC_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


class ConditionRule(BaseModel):
    """Leaf: one tag compared to a value."""

    kind: Literal["condition"] = "condition"
    modality: str = Field(
        default="",
        description="The modality the rule applies to (filled after parse).",
    )
    tag: str = Field(description="The metadata tag the rule applies to.")
    value: str | list[str] = Field(
        description="The value to compare against.",
    )
    comparison: Literal[">", "<", ">=", "<=", "==", "!="] = Field(
        description=(
            "The comparison type. `==`/`!=` interpret the comparison value as regex. "
            "`>`/`<`/`>=`/`<=` interpret the comparison value as a numeric value."
        ),
    )

    def __repr__(self) -> str:
        return _format_rule_repr(self, 0)

    def evaluate(self, row: pd.Series) -> bool:  # noqa: PLR0912, PLR0911
        """Evaluate whether a metadata row (Series indexed by tag names) matches this rule."""
        tag_value = row.get(self.tag)
        if tag_value is None:
            return False
        if pd.isna(tag_value):
            return False

        # Handles list-formatted values
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
                for token in tag_value:
                    for pattern in patterns:
                        if re.search(pattern, token):
                            return True
                return False

            case "!=":
                patterns = self.value
                if isinstance(self.value, str):
                    patterns = [self.value]
                for token in tag_value:
                    for pattern in patterns:
                        if re.search(pattern, token):
                            return False
                return True

            case ">" | "<" | ">=" | "<=":
                op_fn = NUMERIC_OPS[self.comparison]
                if isinstance(self.value, list):
                    msg = f"{self.comparison} comparison only compatible with numeric arguments, not list."
                    raise ValueError(msg)
                for token in tag_value:
                    if token == "" or token is None or pd.isna(token):
                        return False
                    try:
                        if not op_fn(float(token), float(self.value)):
                            return False
                    except ValueError as exc:
                        msg = (
                            f"'{self.comparison}' comparisons only support numeric values."
                            f"\nInput: {self.tag}: {tag_value}, {self.comparison} {self.value}"
                        )
                        raise ValueError(msg) from exc
                return True


class BoolRule(BaseModel):
    """Composite: AND / OR of two subtrees."""

    kind: Literal["bool"] = "bool"
    modality: str = Field(
        default="",
        description="The modality the rule applies to (filled after parse).",
    )
    op: Literal["AND", "OR"]
    left: ConditionRule | BoolRule
    right: ConditionRule | BoolRule

    def __repr__(self) -> str:
        return _format_rule_repr(self, 0)

    def evaluate(self, row: pd.Series) -> bool:
        """Evaluate ``row`` with AND/OR of the two subtrees."""
        if self.op == "AND":
            return self.left.evaluate(row) and self.right.evaluate(row)
        elif self.op == "OR":
            return self.left.evaluate(row) or self.right.evaluate(row)
        else:
            msg = f"Invalid operator: {self.op}"
            raise ValueError(msg)


def _with_modality(
    node: ConditionRule | BoolRule, modality: str
) -> ConditionRule | BoolRule:
    """Attach modality to every node (no mutable transformer state)."""
    if isinstance(node, ConditionRule):
        return node.model_copy(update={"modality": modality})
    return node.model_copy(
        update={
            "modality": modality,
            "left": _with_modality(node.left, modality),
            "right": _with_modality(node.right, modality),
        }
    )


class RuleTransformer(Transformer):
    """Builds a tree without modality; `start` applies modality in one pass."""

    def modality(self, items: list[object]) -> str:
        return str(items[0])

    def condition(self, items: list[object]) -> ConditionRule:
        tag, op, value = items
        value_s = str(value)
        if value_s.startswith('"') and value_s.endswith('"'):
            value_s = value_s[1:-1]
        return ConditionRule(
            tag=str(tag),
            comparison=str(op), # type: ignore
            value=value_s,
        )

    def and_expr(self, items: list[ConditionRule | BoolRule]) -> BoolRule:
        return BoolRule(op="AND", left=items[0], right=items[1])

    def or_expr(self, items: list[ConditionRule | BoolRule]) -> BoolRule:
        return BoolRule(op="OR", left=items[0], right=items[1])

    def value(self, items: list[object]) -> object:
        return items[0]

    def start(self, items: list[object]) -> ConditionRule | BoolRule:
        modality, expr = items
        return _with_modality(expr, str(modality)) # type: ignore


def parse_rule_node(string: str) -> ConditionRule | BoolRule:
    tree = PARSER.parse(string)
    return RuleTransformer().transform(tree)
