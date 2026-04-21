import pandas as pd
from ydata_profiling import ProfileReport
import os

# Load raw data
data_path = "data/raw/superstore_sales.csv"
df = pd.read_csv(data_path, encoding='latin1')

# Generate EDA report
profile = ProfileReport(df, title="Superstore Sales EDA Report", explorative=True)

# Save report
os.makedirs("reports", exist_ok=True)
profile.to_file("reports/eda_report.html")

print("✅ EDA report generated successfully at reports/eda_report.html")
