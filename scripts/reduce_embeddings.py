import os
import pickle

import numpy as np
import pandas as pd
import umap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

reviews_path = os.path.join(BASE_DIR, "outputs", "cleaned_reviews.xlsx")
embeddings_path = os.path.join(BASE_DIR, "models", "embeddings.pkl")
output_path = os.path.join(BASE_DIR, "outputs", "issues", "reviews_with_embeddings.pkl")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Loading cleaned reviews from:", reviews_path)
print("Loading embeddings from:", embeddings_path)

reviews = pd.read_excel(reviews_path)

with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)

if len(reviews) != len(embeddings):
    raise ValueError(
        f"Review count ({len(reviews)}) does not match embeddings count ({len(embeddings)})"
    )

reviews = reviews.reset_index(drop=True)
reviews["embedding"] = list(embeddings)

embeddings_array = np.vstack(reviews["embedding"].values)

reducer = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

print("Fitting UMAP reducer...")
reduced_embeddings = reducer.fit_transform(embeddings_array)

reviews["embedding_reduced"] = list(reduced_embeddings)

reviews.to_pickle(output_path)

print("Saved reviews with embeddings and reduced embeddings to:", output_path)
print("Shape of reduced embeddings:", reduced_embeddings.shape)
