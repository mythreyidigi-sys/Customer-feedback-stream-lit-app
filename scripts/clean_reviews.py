import pandas as pd
import re
import os

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(
    BASE_DIR,
    "data",
    "merged_reviews",
    "all_restaurants_reviews.xlsx"
)

output_file = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_reviews.xlsx"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_excel(input_file)

print("Original Reviews:", len(df))

# Combine restaurant fields from source files with different column casing.
if "restaurant" in df.columns and "Restaurant" in df.columns:
    df["restaurant"] = df["restaurant"].combine_first(df["Restaurant"])
elif "Restaurant" in df.columns and "restaurant" not in df.columns:
    df["restaurant"] = df["Restaurant"]

def normalize_restaurant(name):
    name = str(name).strip().lower()
    if "a2b" in name:
        return "A2B"
    if "geetham" in name:
        return "Geetham"
    if "sangeetha" in name:
        return "Sangeetha"
    if "saravana" in name:
        return "Saravana Bhavan"
    if "annapoorna" in name:
        return "Sree Annapoorna"
    if "vasantha" in name or "vasanta" in name:
        return "Namma Veedu Vasanta Bhavan"
    return name.title()

df["restaurant"] = df["restaurant"].apply(normalize_restaurant)

# =========================
# FIND REVIEW COLUMN
# =========================

review_col = None

for col in df.columns:
    if "review" in col.lower():
        review_col = col
        break

if review_col is None:
    raise Exception("Review column not found!")

print("Review Column Found:", review_col)

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df[review_col] = df[review_col].apply(clean_text)

# =========================
# REMOVE EMPTY / UNKNOWN
# =========================

df = df[df[review_col].notna()]

df = df[df[review_col] != ""]

df = df[df[review_col] != "nan"]

df = df[df[review_col] != "unknown"]

df = df[df[review_col] != "none"]

# =========================
# REMOVE DUPLICATES
# =========================

before = len(df)

df = df.drop_duplicates(subset=[review_col])

after = len(df)

print("Duplicates Removed:", before - after)

# =========================
# SAVE
# =========================

df.to_excel(output_file, index=False)

print("Final Reviews:", len(df))
print("Saved:", output_file)