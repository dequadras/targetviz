"""Sample usage: generate a targetviz report for the Titanic dataset."""

import pandas as pd

import targetviz

# Load the Titanic dataset from a CSV file directly from the web
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic = pd.read_csv(url)

print(f"Loaded Titanic dataset: {titanic.shape[0]} rows, {titanic.shape[1]} columns")
print(f"Columns: {list(titanic.columns)}")
print("Target: Survived\n")

# Generate HTML report
targetviz.targetviz_report(
    titanic,
    target="Survived",
    output_dir="./samples/",
    name_file_out="titanic.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/titanic.html")
