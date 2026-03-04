import pytest

from idc_index import IDCClient

from imgnet.imgnet import ImgNet
from imgnet.query import ValidQuery

from pathlib import Path
import shutil




@pytest.mark.parametrize("collections, modalities, rules, download, result", [
    # Each test is designed to only return one result. 
    (["4D-Lung"], "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, False, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    (["4D-Lung", "Adrenal-ACC-Ki67-Seg"], "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, False, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    ("all", "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"}, False, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.545135956948851752674310887985"]),
    ("all", "CT,RTSTRUCT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069", "RTSTRUCT": "Modality == CT"}, False, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.102533678118509892496905762069"]),

    # Test Download
    (["4D-Lung"], "CT", {"CT": "SeriesInstanceUID == 1.3.6.1.4.1.14519.5.2.1.6834.5010.124741849880980303405787216373"}, True, ["1.3.6.1.4.1.14519.5.2.1.6834.5010.124741849880980303405787216373"])
   
])
def test_imgnet_query(
        collections: str | list[str],
        modalities: str | list[str],
        rules: dict[str, str | list[str]],
        download: bool,
        result: list[str],
        client: IDCClient
) -> None:
    output_dir = Path("./tests/test_dir/ImgNet_query_output")
    query = ValidQuery(collections=collections, modalities=modalities, rules=rules)
    df = ImgNet(output_path=output_dir, client=client).query(valid_query=query, download=download)
    assert df["SeriesInstanceUID"].tolist() == result # check if the query returns the expected series.
    if download:
        assert output_dir.exists(), "Download failed: output dir does not exist."
        assert any(output_dir.iterdir()), "Download failed: No files downloaded."
        shutil.rmtree(output_dir, ignore_errors=True)

