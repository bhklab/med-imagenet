import pandas as pd 
from pathlib import Path
from nbiatoolkit.settings import Settings
# see `all_series.ipynb` for the scratch notebook to 
# generate the `allseries.tsv` file
allseriesdf = pd.read_csv("allseries.tsv", sep="\t", low_memory=False)

supported_modalities = ["SEG"]

# get 5 series from allseriesdf where modality is SEG
collections = allseriesdf[allseriesdf["Modality"].isin(supported_modalities)]["Collection"].unique()
seriesdf = allseriesdf[allseriesdf["Collection"].isin(collections) & (allseriesdf["Modality"].isin(supported_modalities))]

collections = seriesdf["Collection"].unique()
with open("modality_counts.txt", "w") as f:
    f.write(seriesdf["Modality"].value_counts().to_string())

# Expects a .env file in the same directory with the following (double underscore is a separator):
# LOGIN__NBIA_USERNAME="..."
# LOGIN__NBIA_PASSWORD="..."
settings = Settings()
rule all:
    input:
        collection_summaries_rtstructs = expand(
            "results/{collection}_RTSTRUCT_summary.csv",
            collection=collections
        ),
        collection_summaries_seg = expand(
            "results/{collection}_SEG_summary.csv",
            collection=collections
        )


rule collect_summaries:
    input:
        mask_metadata = collect(
            "procdata/{series.Collection}/{series.Modality}/{series.SeriesInstanceUID}.json",
            series = lookup(
                query="Collection == '{collection}' and Modality == '{modality}'",
                within=seriesdf
            )
        )
    output:
        collection_summary = "results/{collection}_{modality}_summary.csv"
    run:
        import json
        # import all the jsons and then convert to pandas csv
        if wildcards.modality == "RTSTRUCT":
            rtstructs = [f for f in input.mask_metadata]
            rtstruct_summaries = []
            for rtstruct in rtstructs:
                with open(rtstruct) as f:
                    rtstruct_summaries.append(json.load(f))
            rtstruct_summaries = pd.DataFrame(rtstruct_summaries)

            allseries_copy = allseriesdf.copy()
            rtstruct_summaries_merged = rtstruct_summaries.merge(
                allseries_copy[["SeriesInstanceUID", "Modality"]],
                left_on="ReferencedSeriesInstanceUID",
                right_on="SeriesInstanceUID",
                how="left",
                suffixes=("", "_referenced")
            ).drop(columns=["SeriesInstanceUID_referenced"])

            # rename Modality_referenced to ReferencedModality
            rtstruct_summaries_merged.rename(columns={"Modality_referenced": "ReferencedModality"}, inplace=True)

            rtstruct_summaries_merged.to_csv(output.collection_summary, index=False)
        elif wildcards.modality == "SEG":
            segs = [f for f in input.mask_metadata]
            seg_summaries = []
            for seg in segs:
                with open(seg) as f:
                    seg_summaries.append(json.load(f))
            seg_summaries = pd.DataFrame(seg_summaries)
            # check if any of the segs have a ReferencedSeriesInstanceUID
            if "ReferencedSeriesInstanceUID" in seg_summaries.columns:
                allseries_copy = allseriesdf.copy()
                seg_summaries_merged = seg_summaries.merge(
                    allseries_copy[["SeriesInstanceUID", "Modality"]],
                    left_on="ReferencedSeriesInstanceUID",
                    right_on="SeriesInstanceUID",
                    how="left",
                    suffixes=("", "_referenced")
                ).drop(columns=["SeriesInstanceUID_referenced"])

                # rename Modality_referenced to ReferencedModality
                seg_summaries_merged.rename(columns={"Modality_referenced": "ReferencedModality"}, inplace=True)

                seg_summaries_merged.to_csv(output.collection_summary, index=False)
            else:
                seg_summaries.to_csv(output.collection_summary, index=False)

rule get_rtstruct_summary:
    output:
        rtstruct_summary = "procdata/{Collection}/RTSTRUCT/{SeriesInstanceUID}.json"
    params:
        NBIA_USERNAME=settings.NBIA_USERNAME,
        NBIA_PASSWORD=settings.NBIA_PASSWORD
    script:
        "scripts/get_rtstruct_summary.py"

rule get_seg_summary:
    output:
        seg_summary = "procdata/{Collection}/SEG/{SeriesInstanceUID}.json"
    params:
        NBIA_USERNAME=settings.NBIA_USERNAME,
        NBIA_PASSWORD=settings.NBIA_PASSWORD
    script:
        "scripts/get_seg_summary.py"