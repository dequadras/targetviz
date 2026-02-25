"""Sample usage: generate a targetviz report for the California Housing dataset."""

import pandas as pd
from sklearn.datasets import fetch_california_housing

import targetviz

# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target

print(f"Loaded California Housing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: MedHouseVal\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="MedHouseVal",
    output_dir="./samples/",
    name_file_out="cal_housing.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/cal_housing.html")
