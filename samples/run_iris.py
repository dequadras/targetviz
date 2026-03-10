"""Sample usage: generate a targetviz report for the Iris dataset."""

import pandas as pd
from sklearn.datasets import load_iris

import targetviz

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["species"] = pd.Categorical.from_codes(data.target, categories=data.target_names)

print(f"Loaded Iris dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: species\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="species",
    output_dir="./samples/",
    name_file_out="iris.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/iris.html")
