"""Sample usage: generate a targetviz report for the UCI Bank Marketing dataset."""

import io
import urllib.request
import zipfile

import pandas as pd

import targetviz

# Load the Bank Marketing dataset from UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
data = urllib.request.urlopen(url).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    with z.open("bank-full.csv") as f:
        df = pd.read_csv(f, sep=";")

print(f"Loaded Bank Marketing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: y (subscribed a term deposit?)\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="y",
    output_dir="./samples/",
    name_file_out="bank_marketing.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/bank_marketing.html")
