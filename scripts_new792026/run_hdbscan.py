import os
import pickle

import hdbscan
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

emb_path = os.path.join(BASE_DIR, "models", "embeddings_reduced.pkl")
review_path = os.path.join(BASE_DIR, "outputs", "cleaned_reviews.xlsx")

print("Loading reduced embeddings from:", emb_path)
with open(emb_path, "rb") as f:
    embeddings = pickle.load(f)

print("Loading reviews from:", review_path)
df = pd.read_excel(review_path)

if len(df) != len(embeddings):
    raise ValueError(f"Review count ({len(df)}) does not match embeddings count ({len(embeddings)})")

if not isinstance(embeddings, np.ndarray):
    embeddings = np.vstack(embeddings)

print("Embeddings shape:", embeddings.shape)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)

cluster_labels = clusterer.fit_predict(embeddings)

df["cluster"] = cluster_labels

print(df["cluster"].value_counts())
print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f"Noise/outlier reviews: {(cluster_labels == -1).sum()}")

output_path = os.path.join(BASE_DIR, "outputs", "issues", "hdbscan_clustered_reviews.xlsx")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_excel(output_path, index=False)
print("Saved clustered reviews to:", output_path)
