import pandas as pd
import numpy as np
from targetviz import targetviz_report

# Define dataset dimensions
n_rows = 10_000  #_000
n_cols = 2

# Generate random data
data = np.random.rand(n_rows, n_cols)

# Create column names
columns = [f'col_{i}' for i in range(n_cols)]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Define target column
target_col = 'col_0'

# Initialize and run TargetViz
print("Initializing TargetViz...")
tv = targetviz_report(data=df, target=target_col, output_dir='./', name_file_out="targetviz_dataset_report.html.zip", pct_outliers=.05)

print(f"TargetViz report saved to 'targetviz_large_dataset_report.html'")
