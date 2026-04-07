# TAKES TOO LONG TO RUN
# import pytest
# from pathlib import Path
# import random

# from imgnet.collections.store import IndexedDatasets

# @pytest.mark.parametrize(
#     "collection, source",
#     [
#         (
#             "MSWAL",
#             "huggingface",
#         ),
#         (
#             "AMOS",
#             "zenodo",
#         ),
#         (
#             "MedicalDecathalon",
#             "s3",
#         ),
#         (
#             "Totalsegmentator",
#             "dropbox",
#         ),
#     ],
# )
# def test_download_from_source(
#     store: IndexedDatasets, 
#     tmp_path: Path,
#     collection: str,
#     source: str
# ) -> None:
#     assert store.source_config(collection).source == source
#     index_df = store.index(collection)

#     # Pick a random file path from the collection's index DataFrame
#     file_path = random.choice(index_df["filepath"].tolist())

#     downloader = store.downloader(collection)
#     downloader.download(tmp_path / collection, instance_ids=[file_path])