import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_folder = os.path.join(BASE_DIR, "Analysis", "data", "raw_reviews_new")

output_file = os.path.join(
    BASE_DIR,
    "data",
    "merged_reviews",
    "all_restaurants_reviews_new.xlsx"
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

            column_aliases = {
                "branch": {"branch"},
                "review": {"review", "reviews"},
                "restaurant": {"restaurant"},
                "date": {"date"},
            }
            for canonical_column, aliases in column_aliases.items():
                matching_columns = [
                    column
                    for column in df.columns
                    if column.strip().lower() in aliases
                ]
                if matching_columns:
                    df[canonical_column] = df[matching_columns].bfill(axis=1).iloc[:, 0]
                    df = df.drop(
                        columns=[
                            column
                            for column in matching_columns
                            if column != canonical_column
                        ]
                    )

            df["source_file"] = os.path.relpath(file_path, BASE_DIR)
            all_dfs.append(df)

if not all_dfs:
    raise ValueError(f"No CSV or XLSX files found in {input_folder}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged_df = pd.concat(all_dfs, ignore_index=True)
if "date" in merged_df.columns:
    merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce").dt.normalize()
with pd.ExcelWriter(output_file, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
    merged_df.to_excel(writer, index=False)
    sheet = writer.sheets["Sheet1"]
    for col_cells in sheet.columns:
        header = col_cells[0].value
        sheet.column_dimensions[col_cells[0].column_letter].width = (
            14 if header == "date" else 40 if header == "review" else 22
        )

print("Merged Reviews:", len(merged_df))
print("Saved:", output_file)
