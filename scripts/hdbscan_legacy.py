import os
import pickle

import hdbscan
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

review_candidates = [
    os.path.join(BASE_DIR, "outputs", "cleaned_reviews.xlsx"),
    os.path.join(BASE_DIR, "outputs", "cleaned_reviews_old.xlsx"),
]
embedding_candidates = [
    os.path.join(BASE_DIR, "models", "embeddings.pkl"),
    os.path.join(BASE_DIR, "models", "embeddings_reduced.pkl"),
    os.path.join(BASE_DIR, "models", "embeddings_old.pkl"),
]

review_path = None
df = None
for candidate in review_candidates:
    if os.path.exists(candidate):
        review_path = candidate
        break

if review_path is None:
    raise FileNotFoundError("No review file found in outputs folder")

print("Loading reviews from:", review_path)
df = pd.read_excel(review_path)

reduced_embeddings = None
emb_path = None
for candidate in embedding_candidates:
    if not os.path.exists(candidate):
        continue
    with open(candidate, "rb") as f:
        embeddings = pickle.load(f)
    if len(df) == len(embeddings):
        reduced_embeddings = embeddings
        emb_path = candidate
        break

if reduced_embeddings is None or emb_path is None:
    raise ValueError(f"No embedding file matches review count ({len(df)})")

print("Loading reduced embeddings from:", emb_path)

if not isinstance(reduced_embeddings, np.ndarray):
    reduced_embeddings = np.vstack(reduced_embeddings)

print("Embeddings shape:", reduced_embeddings.shape)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)

cluster_labels = clusterer.fit_predict(reduced_embeddings)
df["cluster"] = cluster_labels

print(df["cluster"].value_counts())
print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f"Noise/outlier reviews: {(cluster_labels == -1).sum()}")

output_path = os.path.join(BASE_DIR, "outputs", "issues", "hdbscan_legacy_clustered_reviews.xlsx")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)
print("Saved clustered reviews to:", output_path)
