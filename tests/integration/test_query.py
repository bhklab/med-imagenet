import pytest
from imgnet.query import ValidQuery, Rule, ValidQueryError
@pytest.mark.parametrize("collections, modalities, rules, exception", [

    # Test collections validation
    ("all", "all", None, None),
    ("4D-Lung", "all", None, None), # failed
    (["4D-Lung", "Adrenal-ACC-Ki67-Seg"], "all", None, None),
    (["all", "4D-Lung"], "all", None, ValidQueryError),
    ("Unsupported_Collection", "all", None, ValidQueryError),
    (["4D-Lung", "Unsupported_Collection"], "all", None, ValidQueryError),

    # Test Modality Validation
    ("all", ["CT,RTSTRUCT", "CT,SEG"], None, None),
    ("all", 1, None, ValidQueryError),

    # Test Rule Validation
    ("all", "all", {"CT": "TagValue == 1"}, None),
    ("all", "all", {"CT": "TagValue == 1", "MR": "TagValue != 1"}, None),
    ("all", "all", {"CT": ["TagValue == 1", "TagValue > 1"], "MR": "TagValue != 1"}, None),

    ("all", "all", {"CT": "Invalid Rule"}, ValidQueryError),
    ("all", "all", {"CT": "TagValue !<= 1"}, ValidQueryError),
    ("all", "all", {"CT": "TagValue == 1", "MR": "Invalid Rule"}, ValidQueryError),
    ("all", "all", {"CT": "TagValue == 1", "MR": ["TagValue == 1", "Invalid Rule"]}, ValidQueryError)
])
def test_query_collection_validation(
    collections: str | list[str], 
    modalities: str | list[str], 
    rules: dict[str, Rule | list[Rule]], 
    exception: Exception):
    """Test that all Pydantic fields are properly validated."""
    if exception:
        with pytest.raises(exception):
            ValidQuery(collections=collections, modalities=modalities, rules=rules)
    else:
        ValidQuery(collections=collections, modalities=modalities, rules=rules) 

