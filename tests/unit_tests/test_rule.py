import pytest
from imgnet.query import Rule, RuleError

@pytest.mark.parametrize("value, dicom_value, expected", [
    # Single string equality
    ("test", "test", True),
    ("test", "mismatch", False),
    
    # Regex match
    ("te.*", "test", True),
    
    # List input to rule
    (["foo", "bar"], "bar", True),
    (["foo", "bar"], "baz", False),
    
    # DICOM tag string with list-formatted value
    ("bar", "[foo, bar]", True),
    ("qux", "[foo, bar]", False),
    ("bar", "['foo','bar']", True),

    # Value not wrapped in list but list in rule
    (["test"], "test", True),
])
def test_equality_match(value, dicom_value, expected):
    rule = Rule(tag="Tag", value=value, comparison="==")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("rule_value, dicom_value, expected", [
    ("test", "test", False),  # Because "test" matches "test"
    ("nomatch", "test", True),  # Doesn't match, so returns True
    ("te.*", "test", False),
    ("bad.*", "good", True),
])
def test_inequality_match(rule_value, dicom_value, expected):
    rule = Rule(tag="Tag", value=rule_value, comparison="!=")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("rule_value, dicom_value, expected", [
    ("10", "20", True),
    ("10", "5", False),
    ("10", "10", False),
])
def test_numeric_gt(rule_value, dicom_value, expected):
    rule = Rule(tag="Tag", value=rule_value, comparison=">")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("rule_value, dicom_value, expected", [
    ("10", "5", True),
    ("10", "15", False),
    ("10", "10", False),
])
def test_numeric_lt(rule_value, dicom_value, expected):
    rule = Rule(tag="Tag", value=rule_value, comparison="<")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("rule_value, dicom_value, expected", [
    ("10", "10", True),
    ("10", "20", True),
    ("10", "5", False),
])
def test_numeric_gte(rule_value, dicom_value, expected):
    rule = Rule(tag="Tag", value=rule_value, comparison=">=")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("rule_value, dicom_value, expected", [
    ("10", "10", True),
    ("10", "5", True),
    ("10", "20", False),
])
def test_numeric_lte(rule_value, dicom_value, expected):
    rule = Rule(tag="Tag", value=rule_value, comparison="<=")
    assert rule.evaluate({"Tag": dicom_value}) is expected

@pytest.mark.parametrize("comparison", [">", "<", ">=", "<="])
def test_numeric_error_for_non_numeric(comparison):
    rule = Rule(tag="Tag", value="10", comparison=comparison)
    with pytest.raises(RuleError):
        rule.evaluate({"Tag": "non-numeric"})

def test_missing_tag_returns_false():
    rule = Rule(tag="MissingTag", value="something", comparison="==")
    assert rule.evaluate({}) is False

def test_empty_value_returns_false_for_numeric():
    rule = Rule(tag="Tag", value="5", comparison=">")
    assert rule.evaluate({"Tag": ""}) is False

def test_bracketed_string_list_parsing():
    rule = Rule(tag="Tag", value="foo", comparison="==")
    dicom_input = {'Tag': "['foo', 'bar']"}
    assert rule.evaluate(dicom_input) is True

    
