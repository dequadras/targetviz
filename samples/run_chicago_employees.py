"""Sample usage: generate a targetviz report for the Chicago Employees dataset."""

import pandas as pd

import targetviz

# Load the Chicago Employees dataset from the City of Chicago Open Data portal
url = "https://data.cityofchicago.org/api/views/xzkq-xp2w/rows.csv?accessType=DOWNLOAD"
df = pd.read_csv(url)

# Keep only salaried (full-time) employees that have an Annual Salary
df = df.dropna(subset=["Annual Salary"])

print(f"Loaded Chicago Employees dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("Target: Annual Salary\n")

# Generate HTML report
targetviz.targetviz_report(
    df,
    target="Annual Salary",
    output_dir="./samples/",
    name_file_out="chicago_employees.html",
    pct_outliers=0.05,
)

print("\nReport saved to samples/chicago_employees.html")
