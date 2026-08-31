"""
Apply the trained issue classifier to all existing reviews
"""
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned reviews
cleaned_reviews_path = BASE_DIR / "outputs" / "cleaned_reviews.xlsx"
print(f"Loading cleaned reviews from: {cleaned_reviews_path}")
df = pd.read_excel(cleaned_reviews_path)
print(f"Total reviews: {len(df)}")

# Identify review text column
review_col = None
for col in ["review_text", "review", "Reviews"]:
    if col in df.columns:
        review_col = col
        break

if review_col is None:
    raise ValueError(f"Could not find review text column. Available columns: {list(df.columns)}")

print(f"Using column '{review_col}' for review text")

# Rename to review_text for consistency
df["review_text"] = df[review_col]

# Load trained classifier
classifier_path = BASE_DIR / "scripts" / "issue_classifier.joblib"
print(f"\nLoading classifier from: {classifier_path}")
bundle = joblib.load(classifier_path)
model = bundle["model"]
vectorizer = bundle["vectorizer"]

# Apply classifier to all reviews
print("\nClassifying all reviews...")
X = vectorizer.transform(df["review_text"])
df["predicted_issue_category"] = model.predict(X)
df["confidence"] = model.predict_proba(X).max(axis=1)

# Display sample classifications
print("\n=== Sample Classified Reviews ===")
sample_cols = ["restaurant", "review_text", "predicted_issue_category", "confidence"]
print(df[sample_cols].head(10).to_string())

# Issue category distribution
print("\n=== Issue Category Distribution ===")
issue_dist = df["predicted_issue_category"].value_counts().reset_index()
issue_dist.columns = ["Issue Category", "Count"]
issue_dist["Percentage"] = (issue_dist["Count"] / len(df) * 100).round(2)
print(issue_dist.to_string(index=False))

# Restaurant-wise issue breakdown
print("\n=== Issue Distribution by Restaurant ===")
restaurant_issues = pd.crosstab(df["restaurant"], df["predicted_issue_category"])
print(restaurant_issues)

# High confidence vs low confidence
print(f"\n=== Confidence Distribution ===")
print(f"Average confidence: {df['confidence'].mean():.3f}")
print(f"Min confidence: {df['confidence'].min():.3f}")
print(f"Max confidence: {df['confidence'].max():.3f}")

high_conf = (df["confidence"] >= 0.8).sum()
med_conf = ((df["confidence"] >= 0.5) & (df["confidence"] < 0.8)).sum()
low_conf = (df["confidence"] < 0.5).sum()

print(f"High confidence (≥0.8): {high_conf} ({high_conf/len(df)*100:.1f}%)")
print(f"Medium confidence (0.5-0.8): {med_conf} ({med_conf/len(df)*100:.1f}%)")
print(f"Low confidence (<0.5): {low_conf} ({low_conf/len(df)*100:.1f}%)")

# Save classified reviews
output_path = BASE_DIR / "outputs" / "reviews_with_issue_classification.xlsx"
df.to_excel(output_path, index=False)
print(f"\n✓ Saved classified reviews to: {output_path}")

# Save summary statistics
summary_output = BASE_DIR / "outputs" / "issue_classification_summary.xlsx"
with pd.ExcelWriter(summary_output) as writer:
    issue_dist.to_excel(writer, sheet_name="Issue Distribution", index=False)
    restaurant_issues.to_excel(writer, sheet_name="Restaurant Issues")
    
    # High confidence reviews
    high_conf_df = df[df["confidence"] >= 0.8][["restaurant", "review_text", "predicted_issue_category", "confidence"]].head(20)
    high_conf_df.to_excel(writer, sheet_name="High Confidence Examples", index=False)
    
    # Low confidence reviews (potential misclassifications)
    low_conf_df = df[df["confidence"] < 0.5][["restaurant", "review_text", "predicted_issue_category", "confidence"]].head(20)
    low_conf_df.to_excel(writer, sheet_name="Low Confidence Examples", index=False)

print(f"✓ Saved summary statistics to: {summary_output}")

print("\n" + "="*60)
print("Classification Complete!")
print("="*60)
