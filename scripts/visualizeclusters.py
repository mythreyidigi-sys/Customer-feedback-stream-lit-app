import os
import pickle
import numpy as np

import umap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hdbscan_cluster_file = os.path.join(BASE_DIR, "outputs", "issues", "hdbscan_clustered_reviews.xlsx")
kmeans_cluster_file = os.path.join(BASE_DIR, "outputs", "issues", "clustered_reviews.xlsx")
embeddings_file = None
cluster_file = None

if os.path.exists(hdbscan_cluster_file):
    cluster_file = hdbscan_cluster_file
    embeddings_file = os.path.join(BASE_DIR, "models", "embeddings_reduced.pkl")
elif os.path.exists(kmeans_cluster_file):
    cluster_file = kmeans_cluster_file
    embeddings_file = os.path.join(BASE_DIR, "models", "embeddings.pkl")
else:
    raise FileNotFoundError(
        "Could not find clustered reviews file. "
        "Please run scripts/run_hdbscan.py or scripts/cluster_reviews.py first."
    )

print("Loading clustered reviews from:", cluster_file)
df = pd.read_excel(cluster_file)
if "cluster" not in df.columns:
    raise ValueError(f"Expected 'cluster' column in {cluster_file}")

print("Loading embeddings from:", embeddings_file)
with open(embeddings_file, "rb") as f:
    embeddings = pickle.load(f)

if not isinstance(embeddings, np.ndarray):
    embeddings = np.vstack(embeddings)

if len(df) != len(embeddings):
    raise ValueError(
        f"Review count ({len(df)}) does not match embeddings count ({len(embeddings)})"
    )

cluster_labels = df["cluster"].to_numpy()

viz_reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
viz_embeddings = viz_reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1], c=cluster_labels, cmap="tab20", s=8)
plt.title("Customer Complaint Clusters (2D projection)")
plt.savefig("cluster_visualization.png", dpi=150)
plt.show()


def get_representative_samples(df, reduced_embeddings, cluster_id, n_samples=15):
    """Return the n_samples reviews closest to the cluster centroid."""
    mask = df["cluster"] == cluster_id
    cluster_points = reduced_embeddings[mask.to_numpy()]
    cluster_indices = df[mask].index.to_numpy()

    centroid = cluster_points.mean(axis=0).reshape(1, -1)
    _, _ = pairwise_distances_argmin_min(cluster_points, centroid)

    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    order = np.argsort(distances)[:n_samples]

    sample_indices = cluster_indices[order]
    sample_df = df.loc[sample_indices].copy()

    if "review_text" not in sample_df.columns and "review" in sample_df.columns:
        sample_df["review_text"] = sample_df["review"]
    if "rating" not in sample_df.columns:
        sample_df["rating"] = None

    keep_cols = ["cluster", "restaurant", "branch", "review_text", "rating"]
    keep_cols = [col for col in keep_cols if col in sample_df.columns]
    return sample_df[keep_cols].to_dict("records")

unique_clusters = sorted(set(cluster_labels) - {-1})
cluster_samples = {
    cid: get_representative_samples(df, embeddings, cid, n_samples=15)
    for cid in unique_clusters
}

print(f"Extracted representative samples for {len(cluster_samples)} clusters.")


