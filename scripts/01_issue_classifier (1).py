"""
01_issue_classifier.py
------------------------
Module 6, Component 1 -- Real-time Root-Cause / Issue-Category Classifier.

Turns the HDBSCAN + Groq clustering output into a reusable, real-time
SUPERVISED classifier: train once on the 5,909 already-labeled reviews, then
tag every new incoming review into one of the 10 issue categories instantly,
without re-running HDBSCAN on every new batch of reviews.

Usage
-----
    python 01_issue_classifier.py

Swap-in for real data
----------------------
Replace `load_or_generate_reviews()` with:
    df = pd.read_excel("cleaned_reviews.xlsx")   # your real labeled export
as long as it has the columns documented in sample_data.py.

If you already have Sentence-Transformer embeddings saved from
generate_embeddings.py (e.g. an `embeddings.npy` aligned row-for-row with the
dataframe), pass `embeddings_path="embeddings.npy"` to `train()` --  this is
strongly preferred over the TF-IDF fallback below, since it keeps the
classifier consistent with the semantic space HDBSCAN clustered on.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import with flexible naming
try:
    from sample_data import load_or_generate_reviews
except ImportError:
    from importlib import import_module
    sample_data_module = import_module('sample_data (1)')
    load_or_generate_reviews = sample_data_module.load_or_generate_reviews

MODEL_OUT = "issue_classifier.joblib"


def build_features(df, embeddings_path=None):
    """Return (X, feature_pipeline_or_None).

    If embeddings_path is given, X is the raw embedding matrix and the
    feature_pipeline is None (the caller re-embeds new text the same way
    generate_embeddings.py did). Otherwise falls back to TF-IDF, fit here.
    """
    if embeddings_path:
        X = np.load(embeddings_path)
        return X, None
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(df["review_text"])
    return X, tfidf


def train(df=None, embeddings_path=None, model_out=MODEL_OUT):
    df = df if df is not None else load_or_generate_reviews()
    labeled = df.dropna(subset=["issue_category"]).reset_index(drop=True)
    print(f"Training on {len(labeled)} labeled reviews across "
          f"{labeled['issue_category'].nunique()} issue categories.")

    X, vectorizer = build_features(labeled, embeddings_path)
    y = labeled["issue_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_model, best_f1 = None, None, -1
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        print(f"\n=== {name} (macro-F1 = {macro_f1:.3f}) ===")
        print(classification_report(y_test, preds, zero_division=0))
        if macro_f1 > best_f1:
            best_name, best_model, best_f1 = name, clf, macro_f1

    print(f"\nSelected model: {best_name} (macro-F1 = {best_f1:.3f})")

    joblib.dump({"model": best_model, "vectorizer": vectorizer,
                 "model_name": best_name, "classes": sorted(y.unique())}, model_out)
    print(f"Saved trained classifier -> {model_out}")
    return best_model, vectorizer, best_name


def classify_new_reviews(review_texts, model_path=MODEL_OUT, embeddings=None):
    """Real-time scoring entry point.

    Pass a list/Series of raw review strings (TF-IDF path) OR a precomputed
    embedding matrix aligned to review_texts (embeddings path).
    Returns a DataFrame with predicted issue_category + confidence.
    """
    bundle = joblib.load(model_path)
    model, vectorizer = bundle["model"], bundle["vectorizer"]

    if embeddings is not None:
        X = embeddings
    else:
        if vectorizer is None:
            raise ValueError("Model was trained on embeddings; pass `embeddings=`.")
        X = vectorizer.transform(review_texts)

    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1)
    else:
        confidence = np.full(len(preds), np.nan)

    return pd.DataFrame({
        "review_text": list(review_texts),
        "predicted_issue_category": preds,
        "confidence": confidence,
    })


if __name__ == "__main__":
    df = load_or_generate_reviews()
    train(df)

    # --- demo: score a few brand-new, unseen reviews in real time ---
    new_reviews = [
        "the portion size was way too small for what we paid",
        "waiter never came to check on us, food took an hour",
        "loved the new dishes on the menu, quick service too",
        "restroom was dirty and tables were sticky",
    ]
    print("\n--- Real-time classification demo ---")
    print(classify_new_reviews(new_reviews).to_string(index=False))
