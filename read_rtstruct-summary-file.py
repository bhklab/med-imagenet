from ast import literal_eval

import pandas as pd

df = pd.read_csv("results/4D-Lung_RTSTRUCT_summary.csv")

list_cols = ["OriginalROINames", "ExtractableROINames", "ReferencedSOPInstanceUIDs"]

for col in list_cols:
    if col in df.columns:
        df[col] = df[col].apply(literal_eval)
print(df.head())
