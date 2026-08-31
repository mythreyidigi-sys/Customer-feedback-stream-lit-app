import pandas as pd

df = pd.read_csv("sree_annapoorna_reviews_tamilnadu.csv")

df.to_excel(
    "sree_annapoorna_reviews_tamilnadu.xlsx",
    index=False
)

print("Excel file created successfully!")