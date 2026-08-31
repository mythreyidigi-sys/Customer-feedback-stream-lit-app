import pandas as pd

# 👇 change this to your actual CSV file name
csv_file = "sangeetha_reviews.csv"

# 👇 output Excel file name
xlsx_file = "sangeetha_reviews.xlsx"

# read CSV file
df = pd.read_csv(csv_file)

# (optional) clean data
df = df.drop_duplicates()
df = df.dropna()

# convert to Excel
df.to_excel(xlsx_file, index=False)

print("✅ Excel file created successfully:", xlsx_file)