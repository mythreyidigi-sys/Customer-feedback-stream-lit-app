import pandas as pd

# 👇 your A2B CSV file name
csv_file = "a2b_reviews_tamilnadu.csv"

# 👇 output Excel file name
xlsx_file = "a2b_reviews_tamilnadu.xlsx"

# read CSV file
df = pd.read_csv(csv_file)

# (optional cleaning)
df = df.drop_duplicates()
df = df.dropna()

# convert to Excel
df.to_excel(xlsx_file, index=False)

print("✅ A2B Excel file created successfully:", xlsx_file)