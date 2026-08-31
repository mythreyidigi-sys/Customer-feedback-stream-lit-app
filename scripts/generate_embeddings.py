import pandas as pd
import os
import pickle

from sentence_transformers import SentenceTransformer

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned_reviews.xlsx"
)

model_folder = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(model_folder, exist_ok=True)

# =========================
# LOAD DATA
# =========================

df = pd.read_excel(input_file)

print("Reviews Loaded:", len(df))

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

# =========================
# LOAD MODEL
# =========================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# GENERATE EMBEDDINGS
# =========================

reviews = df[review_col].astype(str).tolist()

embeddings = model.encode(
    reviews,
    show_progress_bar=True
)

# =========================
# SAVE EMBEDDINGS
# =========================

embedding_file = os.path.join(
    model_folder,
    "embeddings.pkl"
)

with open(embedding_file, "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings Saved!")
print("Location:", embedding_file)
print("Total Embeddings:", len(embeddings))