import pandas as pd

df = pd.read_csv("vasantha_bhavan_reviews_tamilnadu.csv")

df.to_excel(
    "vasantha_bhavan_reviews_tamilnadu.xlsx",
    index=False
)

print("Excel file created successfully!")