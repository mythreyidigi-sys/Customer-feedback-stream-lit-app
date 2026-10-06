import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "merged_reviews", "all_restaurants_reviews_new.xlsx")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "cleaned_reviews.xlsx")


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
        return "Namma Veedu Vasantha Bhavan"
    return name.title()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_reviews_dataframe(df):
    """Normalize restaurant and review text, remove invalid rows and duplicate reviews."""
    df = df.copy()
    original_count = len(df)

    restaurant_columns = [column for column in df.columns
                          if str(column).strip().casefold() == "restaurant"]
    if restaurant_columns:
        restaurant_names = df[restaurant_columns[0]]
        for column in restaurant_columns[1:]:
            restaurant_names = restaurant_names.combine_first(df[column])
        df["restaurant"] = restaurant_names.apply(normalize_restaurant)

    review_column = next((column for column in df.columns if "review" in str(column).lower()), None)
    if review_column is None:
        raise ValueError("Review column not found.")

    df[review_column] = df[review_column].apply(clean_text)
    valid_reviews = df[review_column].notna() & ~df[review_column].isin(["", "nan", "unknown", "none"])
    df = df[valid_reviews].copy()
    invalid_removed = original_count - len(df)

    before_deduplication = len(df)
    df = df.drop_duplicates(subset=[review_column]).copy()
    stats = {
        "original": original_count,
        "invalid_removed": invalid_removed,
        "duplicates_removed": before_deduplication - len(df),
        "final": len(df),
        "review_column": review_column,
    }
    return df, stats


def clean_reviews_file(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_excel(input_file)
    cleaned, stats = clean_reviews_dataframe(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cleaned.to_excel(output_file, index=False)
    return cleaned, stats


if __name__ == "__main__":
    cleaned_reviews, cleaning_stats = clean_reviews_file()
    print("Original Reviews:", cleaning_stats["original"])
    print("Review Column Found:", cleaning_stats["review_column"])
    print("Invalid Reviews Removed:", cleaning_stats["invalid_removed"])
    print("Duplicates Removed:", cleaning_stats["duplicates_removed"])
    print("Final Reviews:", cleaning_stats["final"])
    print("Saved:", OUTPUT_FILE)