import asyncio
import json
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZipFile

from imgtools.dicom.input import load_seg_dcm
from nbiatoolkit import nbia
from nbiatoolkit.nbia import FailedQueryError

if TYPE_CHECKING:
    from snakemake.script import snakemake

client = nbia.NBIAClient(
    username=snakemake.params["NBIA_USERNAME"],
    password=snakemake.params["NBIA_PASSWORD"],
    disable_progress_bar=True,
)


try:
    sop = client.getSOPInstanceUIDs(
        SeriesInstanceUID=snakemake.wildcards["SeriesInstanceUID"]
    )

    raw_images = client.downloadImage(sop)
except (FailedQueryError, AssertionError):
    task = client._downloadSeries(
        params={"SeriesInstanceUID": snakemake.wildcards["SeriesInstanceUID"]}
    )
    raw_series = asyncio.run(task)
    # the raw_series is a zip file
    # without writing to temp directory, lets get the list of files
    with ZipFile(BytesIO(raw_series)) as zf:
        files = [f for f in zf.namelist() if f.endswith(".dcm")]
        if len(files) == 1:
            raw_images = zf.read(files[0])
        else:
            errmsg = f"Expected a single image in zip, got {len(files)}"
            errdict = {
                "errmsg" : errmsg,
                "Found files": files,
            }
            series_metadata = asyncio.run(
                client._getSeries(params={"SeriesInstanceUID": snakemake.wildcards["SeriesInstanceUID"]})
            )
            if len(series_metadata) != 1:
                raise ValueError(
                    "idk man something happened"
                )
            errdict.update(series_metadata[0])

            with open(snakemake.output[0], "w") as f:
                json.dump(errdict, f, indent=4)
            exit(0)


seg = load_seg_dcm(raw_images)

metadata = {
    "PatientID": seg.PatientID,
    "StudyInstanceUID": seg.StudyInstanceUID,
    "SeriesInstanceUID": seg.SeriesInstanceUID,
    "Modality": seg.Modality,
    "OriginalROINames": [],
}


for segment in seg.SegmentSequence:
    metadata["OriginalROINames"].append(segment.SegmentLabel)

if hasattr(seg, "ReferencedSeriesSequence"):
    if len(seg.ReferencedSeriesSequence) != 1:
        raise ValueError(
            f"Expected a single ReferencedSeriesSequence, got {len(seg.ReferencedSeriesSequence)}"
        )

    metadata["ReferencedSeriesInstanceUID"] = seg.ReferencedSeriesSequence[0].get(
        "SeriesInstanceUID", ""
    )

    referenced_sop_uids = []

    if hasattr(seg.ReferencedSeriesSequence[0], "ReferencedInstanceSequence"):
        for ref in seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence:
            referenced_sop_uids.append(ref.ReferencedSOPInstanceUID)

    metadata["ReferencedSOPInstanceUIDs"] = referenced_sop_uids
elif hasattr(seg, "SourceImageSequence"):
    referenced_sop_uids = []
    for source in seg.SourceImageSequence:
        referenced_sop_uids.append(source.get("ReferencedSOPInstanceUID", ""))
    metadata["ReferencedSOPInstanceUIDs"] = referenced_sop_uids
    metadata["ReferencedSeriesInstanceUID"] = ""

with open(snakemake.output[0], "w") as f:
    json.dump(metadata, f, indent=4)

import pathlib

if not pathlib.Path(snakemake.output[0]).exists():
    raise FileNotFoundError(f"Output file {snakemake.output[0]} not found")
