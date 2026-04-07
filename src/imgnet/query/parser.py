import re

from imgnet.query.models import Rule, RulesValidationParsingError

SUPPORTED_COMPARISONS = {"=", "==", "<", ">", "<=", ">=", "!="}


def parse_rule(rule: str) -> Rule:
    """Parse a rule string into a `Rule` instance.

    Accepted syntax:
    ``'<tag> <comparison> <value>'``.
    """

    rule_parts = rule.split(" ", 2)
    if len(rule_parts) != 3:
        raise RulesValidationParsingError("Invalid rule syntax.")

    tag, comparison, raw_value = rule_parts
    if comparison not in SUPPORTED_COMPARISONS:
        msg = (
            f"{comparison} is not a supported comparison type."
            f"\nSupported comparison types: {SUPPORTED_COMPARISONS}"
        )
        raise RulesValidationParsingError(msg)

    value: list[str] | str
    if raw_value.startswith("["):
        matches = re.findall(r"""(['"])(.*?)\1""", raw_value)
        value = [m[1] for m in matches]
    else:
        value = raw_value.strip("'\"")

    return Rule(tag=tag, value=value, comparison=comparison)
