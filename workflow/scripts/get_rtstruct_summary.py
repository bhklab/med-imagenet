import asyncio
import json
from io import BytesIO
from typing import TYPE_CHECKING
from zipfile import ZipFile

from imgtools.modules.structureset import (  # type: ignore
    StructureSet,
    roi_names_from_dicom,
)
from nbiatoolkit import nbia
from nbiatoolkit.nbia import FailedQueryError

if TYPE_CHECKING:
    from snakemake.script import snakemake

client = nbia.NBIAClient(
    username=snakemake.params["NBIA_USERNAME"],
    password=snakemake.params["NBIA_PASSWORD"],
    disable_progress_bar=True
)


try:
    sop = client.getSOPInstanceUIDs(
        SeriesInstanceUID=snakemake.wildcards["SeriesInstanceUID"]
    )

    raw_images = client.downloadImage(sop)
except FailedQueryError:
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
            raise ValueError(
                f"Expected a single image in zip, got {len(files)}, {files}"
            )


# Ensure raw_images is a list of bytes
if isinstance(raw_images, bytes):
    pass
elif isinstance(raw_images, list):
    if len(raw_images) == 1:
        raw_images = raw_images[0]
    else:
        raise ValueError(f"Expected a single image, got {len(raw_images)}")
else:
    raise ValueError(f"Unexpected type for raw_images: {type(raw_images)}")

rtdcm = StructureSet._load_rtstruct_data(raw_images)
rtstruct = StructureSet.from_dicom(raw_images, suppress_warnings=True)
original_roi_names = roi_names_from_dicom(rtdcm)

metadata = rtstruct.metadata

metadata["OriginalROINames"] = original_roi_names
metadata["ExtractableROINames"] = rtstruct.roi_names

with open(snakemake.output[0], "w") as f:
    json.dump(metadata, f, indent=4)
