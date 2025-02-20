import pandas as pd 
from pathlib import Path
from nbiatoolkit.settings import Settings
from ast import literal_eval
from collections import Counter
import json
# see `all_series.ipynb` for the scratch notebook to 
# generate the `allseries.tsv` file
allseriesdf = pd.read_csv("allseries.tsv", sep="\t", low_memory=False)

supported_modalities = ["SEG", "RTSTRUCT"]

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
        expand("results_summaries/0_{modality}_all_summary.csv", modality=supported_modalities)

rule collect_results:
    input:
        collect(
            "results/{series.Collection}/{series.Modality}/{series.Collection}_{series.Modality}_summary.csv",
            series = lookup(
                query="Modality == '{modality}'",
                within=seriesdf[['Collection', 'Modality']].drop_duplicates()
            )
        )
    output:
        collected = "results_summaries/0_{modality}_all_summary.csv",
        summary_refs = "results_summaries/1_{modality}_summary_refs.md",
        roi_name_counts = "results_summaries/2_{modality}_roi_name_counts.json"
    run:
        # essentially concatenate all the csvs
        input_files = [Path(f) for f in input]
        dfs = []
        for f in input_files:
            try:
                df = pd.read_csv(f)
                collection, modality = f.parts[-3:-1]
                df["Collection"] = collection
                dfs.append(df)
            except pd.errors.EmptyDataError:
                pass

        dfs = [df for df in dfs if not df.empty]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(output.collected, index=False)

        df_counts = df.value_counts(subset=['Collection', 'ReferencedModality']).reset_index(name='count')
        df_counts['ReferencedModality'] = df_counts['ReferencedModality'].fillna('Unknown')
        df_pivot = df_counts.pivot(index='Collection', columns='ReferencedModality', values='count').fillna(0)
        # Ensure integer format
        df_pivot = df_pivot.astype(int)
        df_pivot.to_markdown(output.summary_refs)

        collection_roi_name_counts = {} 

        #         # we want to groupby collection and then for each row, get the ExtractableROINames and then count the number of ROIs
        #         # so we get a count of ROINames for each collection
        #         # goal:
        #         # {
        #         #    "collection1": {
        #         #        "roi1": 10,
        #         #        "roi2": 5
        #         #    },
        #         # ...
        for collection, subset_df in df.groupby('Collection'):
            match wildcards.modality:
                case "RTSTRUCT":
                    roi_names = subset_df.ExtractableROINames.apply(literal_eval).explode()
                case "SEG":
                    roi_names = subset_df.OriginalROINames.apply(literal_eval).explode()

            c = roi_names.value_counts().to_dict()
            collection_roi_name_counts[collection] = c

        with open(output.roi_name_counts, "w") as f:
            json.dump(collection_roi_name_counts, f, indent=4)

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
        collection_summary = "results/{collection}/{modality}/{collection}_{modality}_summary.csv",
        errors = "results/{collection}/{modality}/{collection}_{modality}_errors.csv"
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

            # empty error file for RTSTRUCT
            with open(output.errors, "w") as f:
                f.write("")

        elif wildcards.modality == "SEG":
            segs = [f for f in input.mask_metadata]
            seg_summaries = []
            seg_errors = []
            for seg in segs:
                with open(seg) as f:
                    loaded_meta = json.load(f)
                    if "errmsg" in loaded_meta:
                        seg_errors.append(loaded_meta)
                    else:
                        seg_summaries.append(loaded_meta)
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
                # if there is no data at all, just touch the file
                if seg_summaries.empty:
                    Path(output.collection_summary).touch()
                else:
                    # add empty column for ReferencedModality and ReferencedSeriesInstanceUID
                    seg_summaries["ReferencedModality"] = "Unknown"
                    seg_summaries["ReferencedSeriesInstanceUID"] = "Unknown"
                    seg_summaries.to_csv(output.collection_summary, index=False)
            
            seg_errors_df = pd.DataFrame(seg_errors)
            seg_errors_df.to_csv(output.errors, index=False)
