"""Sample usage: generate a targetviz report for the Wine Quality dataset."""

import pandas as pd

import targetviz

# Load the Wine Quality dataset (red wine) from UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

print(f"Loaded Wine Quality dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: quality\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="quality",
    output_dir="./samples/",
    name_file_out="wine_quality.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/wine_quality.html")
