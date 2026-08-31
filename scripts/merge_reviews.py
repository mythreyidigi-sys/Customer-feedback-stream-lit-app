import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_folder = os.path.join(BASE_DIR, "data", "raw_reviews")

output_file = os.path.join(
    BASE_DIR,
    "data",
    "merged_reviews",
    "all_restaurants_reviews.xlsx"
)

all_dfs = []

for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".xlsx", ".csv")):
            file_path = os.path.join(root, file)
            print("Reading:", file_path)

            try:
                if file.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception as exc:
                print(f"Skipping {file_path}: {exc}")
                continue

            df["source_file"] = os.path.relpath(file_path, BASE_DIR)
            all_dfs.append(df)

if not all_dfs:
    raise ValueError(f"No CSV or XLSX files found in {input_folder}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_excel(output_file, index=False)

print("Merged Reviews:", len(merged_df))
print("Saved:", output_file)
