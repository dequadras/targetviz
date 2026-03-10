"""Sample usage: generate a targetviz report for the Census Income (Adult) dataset."""

import pandas as pd

import targetviz

# Load the Census Income dataset from UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
df = pd.read_csv(url, header=None, names=columns, skipinitialspace=True)

# Clean missing values encoded as '?'
df = df.replace("?", pd.NA)

print(f"Loaded Census Income dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: income\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="income",
    output_dir="./samples/",
    name_file_out="census.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/census.html")
