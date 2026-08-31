import os
import pickle

import numpy as np
import umap

# Load embeddings saved by scripts/generate_embeddings.py (a numpy array or list)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb_path = os.path.join(BASE_DIR, "models", "embeddings.pkl")
out_path = os.path.join(BASE_DIR, "models", "embeddings_reduced.pkl")

print("Loading embeddings from:", emb_path)
with open(emb_path, "rb") as f:
    embeddings = pickle.load(f)

# Ensure embeddings is a 2D numpy array
emb_arr = np.vstack(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
if emb_arr.ndim != 2:
    raise ValueError(f"Loaded embeddings have unexpected shape: {getattr(emb_arr, 'shape', None)}")

print("Embeddings shape:", emb_arr.shape)

reducer = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
)

print("Fitting UMAP reducer...")
reduced_embeddings = reducer.fit_transform(emb_arr)

print("Reduced embeddings shape:", reduced_embeddings.shape)

with open(out_path, "wb") as f:
    pickle.dump(reduced_embeddings, f)

print("Saved reduced embeddings to:", out_path)

