import pandas as pd 
from pathlib import Path
from nbiatoolkit.settings import Settings
allseriesdf = pd.read_csv("allseries.tsv", sep="\t", low_memory=False)
collections = allseriesdf[allseriesdf["Modality"] == "RTSTRUCT"]["Collection"].unique()

collections = collections[:9]
seriesdf = allseriesdf[allseriesdf["Collection"].isin(collections) & (allseriesdf["Modality"] == "RTSTRUCT")]
# subsetdf = seriesdf[seriesdf["Collection"].isin(collections_with_rtstructs) & (seriesdf["Modality"] == "RTSTRUCT")]


# sampledf = subsetdf.sample(10)
# sampledf.to_csv("sample_series.tsv", sep="\t", index=False)
# sampledf = pd.read_csv("sample_series.tsv", sep="\t", low_memory=False)
# collections = sampledf["Collection"].unique()

settings = Settings()

rule all:
    input:
        "results/summary.csv"

rule summarize_data:
    input:
        collection_summaries = expand(
            "results/{collection}_RTSTRUCT_summary.csv",
            collection=collections
        )
    output:
        "results/summary.csv"
    run:
        # combine all the collection summaries into one
        dfs = []
        for collection_summary in input.collection_summaries:
            dfs.append(pd.read_csv(collection_summary))

        summary = pd.concat(dfs)
        summary.to_csv(output[0], index=False)

rule collect_rtstruct_summaries:
    input:
        rtstructs = collect(
            "procdata/{series.Collection}/{series.Modality}/{series.SeriesInstanceUID}.json",
            series = lookup(
                query="Collection == '{collection}'",
                within=seriesdf
            )
        )
    output:
        collection_summary = "results/{collection}_RTSTRUCT_summary.csv"
    run:
        import json
        # import all the jsons and then convert to pandas csv
        rtstructs = [f for f in input.rtstructs]
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

rule get_rtstruct_summary:
    output:
        rtstruct_summary = "procdata/{Collection}/RTSTRUCT/{SeriesInstanceUID}.json"
    params:
        NBIA_USERNAME=settings.NBIA_USERNAME,
        NBIA_PASSWORD=settings.NBIA_PASSWORD
    script:
        "scripts/get_rtstruct_summary.py"