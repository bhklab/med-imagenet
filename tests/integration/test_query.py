import pytest
from imgnet.query import ValidQuery, Rule, ValidQueryError, CollectionsValidationError
from imgnet.collections.store import IndexedDatasets


@pytest.mark.parametrize("collections, modalities, rules, exception", [
    # Valid construction (type-level validation only)
    ("all", "all", None, None),
    ("4D-Lung", "all", None, None),
    (["4D-Lung", "Adrenal-ACC-Ki67-Seg"], "all", None, None),

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
def test_query_field_validation(
    collections: str | list[str], 
    modalities: str | list[str], 
    rules: dict[str, Rule | list[Rule]], 
    exception: Exception) -> None:
    """Test that Pydantic field validation catches type/format errors at construction time."""
    if exception:
        with pytest.raises(exception):
            ValidQuery(collections=collections, modalities=modalities, rules=rules)
    else:
        ValidQuery(collections=collections, modalities=modalities, rules=rules) 


@pytest.mark.parametrize("collections, expected_exception", [
    ("Unsupported_Collection", CollectionsValidationError),
    (["4D-Lung", "Unsupported_Collection"], CollectionsValidationError),
    (["all", "4D-Lung"], CollectionsValidationError),
])
def test_query_collection_membership_validation(
    collections: str | list[str],
    expected_exception: type[Exception],
    store: IndexedDatasets,
) -> None:
    """Collection membership against the store is validated at process() time."""
    query = ValidQuery(collections=collections, modalities="all", rules=None)
    with pytest.raises(expected_exception):
        query.process(store)


@pytest.mark.parametrize("collections, modalities, rules, result", [
    # Each test is designed to only return one result. 
    (["4D-Lung"], "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    (["4D-Lung", "Adrenal-ACC-Ki67-Seg"], "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    ("all", "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    ("all", "CT,RTSTRUCT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069", "RTSTRUCT": "Modality == CT"}, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069"])    
])
def test_query_process(
        collections: str | list[str],
        modalities: str | list[str],
        rules: dict[str, Rule | list[Rule]],
        result: list[str],
        store: IndexedDatasets,
) -> None:
    df = ValidQuery(collections=collections, modalities=modalities, rules=rules).process(store)
    assert df["SeriesInstanceUID"].tolist() == result
