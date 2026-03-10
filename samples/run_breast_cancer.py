"""Sample usage: generate a targetviz report for the Breast Cancer Wisconsin dataset."""

import pandas as pd
from sklearn.datasets import load_breast_cancer

import targetviz

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["diagnosis"] = pd.Categorical.from_codes(data.target, categories=data.target_names)

print(f"Loaded Breast Cancer dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: diagnosis\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="diagnosis",
    output_dir="./samples/",
    name_file_out="breast_cancer.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/breast_cancer.html")
