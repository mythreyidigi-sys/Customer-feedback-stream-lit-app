import pandas as pd

df = pd.read_csv("saravana_bhavan_reviews_tamilnadu.csv")

df.to_excel(
    "saravana_bhavan_reviews_tamilnadu.xlsx",
    index=False
)

print("Excel file created successfully!")