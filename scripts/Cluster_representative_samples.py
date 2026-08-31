import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hdbscan_cluster_file = os.path.join(BASE_DIR, "outputs", "issues", "hdbscan_clustered_reviews.xlsx")
kmeans_cluster_file = os.path.join(BASE_DIR, "outputs", "issues", "clustered_reviews.xlsx")

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

restaurant_columns = [
    column for column in ("restaurant", "Restaurant") if column in df.columns
]
if not restaurant_columns:
    raise ValueError(f"No restaurant column found in {cluster_file}")

df["restaurant"] = df[restaurant_columns[0]]
for column in restaurant_columns[1:]:
    df["restaurant"] = df["restaurant"].fillna(df[column])

restaurant_aliases = {
    "a2b - adyar ananda bhavan": "A2B",
    "geetham veg restaurant": "Geetham",
    "hotel saravana bhavan": "Saravana Bhavan",
    "saravanabhavan": "Saravana Bhavan",
    "sree annapoorna sree gowrishankar": "Sree Annapoorna",
    "sri annapoorna": "Sree Annapoorna",
    "sree annapoorna": "Sree Annapoorna",
    "annapoorna": "Sree Annapoorna",
    "annapoorna hot veg restaurant": "Sree Annapoorna",
    "chennai annapoorna": "Sree Annapoorna",
    "house of annapoorna": "Sree Annapoorna",
    "sangeetha veg restaurant": "Sangeetha",
    "sangeetha's desi mane": "Sangeetha",
    "vasanta bhavan": "Namma Veedu Vasanta Bhavan",
    "vasantha bhavan": "Namma Veedu Vasanta Bhavan",
    "vasanthabhavan": "Namma Veedu Vasanta Bhavan",
}
df["restaurant"] = (
    df["restaurant"].astype("string").str.strip()
    .apply(lambda value: restaurant_aliases.get(str(value).lower(), value))
)

print("Loading embeddings from:", embeddings_file)
with open(embeddings_file, "rb") as f:
    embeddings = pickle.load(f)

if not isinstance(embeddings, np.ndarray):
    embeddings = np.vstack(embeddings)

if len(df) != len(embeddings):
    raise ValueError(
        f"Review count ({len(df)}) does not match embeddings count ({len(embeddings)})"
    )


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
    sample_df = sample_df[keep_cols]

    sample_df = sample_df.reset_index(drop=True)
    sample_df["distance_to_centroid"] = distances[order]
    return sample_df


cluster_labels = df["cluster"].to_numpy()
unique_clusters = sorted(set(cluster_labels) - {-1})

cluster_samples = []
for cid in unique_clusters:
    sample_df = get_representative_samples(df, embeddings, cid, n_samples=15)
    cluster_samples.append(sample_df)

if cluster_samples:
    output_path = os.path.join(BASE_DIR, "outputs", "issues", "cluster_representative_samples.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.concat(cluster_samples, ignore_index=True).to_csv(output_path, index=False)
    print(f"Saved representative samples for {len(unique_clusters)} clusters to:", output_path)
else:
    print("No non-noise clusters were found.")
