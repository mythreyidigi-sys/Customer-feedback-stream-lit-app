"""OPTIONAL: re-label complaint reviews with dynamic clusters (your dissertation method).

    Sentence-Transformer (all-MiniLM-L6-v2) embeddings -> HDBSCAN -> Groq names each cluster
    -> reviews.issue is updated in empathy.db (issue_source = 'hdbscan').

Run from the project folder:
    python tools/discover_issues.py                  # MiniLM embeddings (pip install sentence-transformers)
    python tools/discover_issues.py --embedder tfidf # no model download; quick test
    python tools/discover_issues.py --dry-run        # print clusters, don't write to the DB

Reviews that HDBSCAN marks as noise keep their keyword label.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import db  # noqa: E402
from engine.classifier import urgency_of  # noqa: E402
from engine.config import POSITIVE_ISSUE  # noqa: E402
from engine.replies import _groq  # noqa: E402


def embed(texts, how):
    if how == "minilm":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    x = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2)).fit_transform(texts)
    return normalize(TruncatedSVD(n_components=min(100, x.shape[1] - 1), random_state=42).fit_transform(x))


def cluster(emb, min_cluster_size):
    from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3 (or: import hdbscan)
    # UMAP first (pip install umap-learn) gives tighter clusters on 384-d vectors; used when available.
    try:
        import umap
        emb = umap.UMAP(n_neighbors=15, n_components=10, metric="cosine", random_state=42).fit_transform(emb)
    except ImportError:
        pass
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, metric="euclidean").fit_predict(emb)


def top_terms(texts, n=6):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    x = cv.fit_transform(texts)
    idx = np.asarray(x.sum(0)).ravel().argsort()[::-1][:n]
    return [cv.get_feature_names_out()[i] for i in idx]


def name_cluster(samples, terms):
    prompt = ("These customer reviews of Chennai vegetarian restaurants belong to one complaint cluster.\n"
              + "\n".join(f"- {s[:250]}" for s in samples)
              + f"\nFrequent terms: {', '.join(terms)}\n"
              "Return JSON only: {\"label\": \"2-5 word business issue name, e.g. 'Slow Service & Staff Negligence'\"}")
    out = _groq(prompt, max_tokens=60, temperature=0)
    if out:
        try:
            return json.loads(out[out.find("{"): out.rfind("}") + 1])["label"]
        except (ValueError, KeyError):
            pass
    return " / ".join(t.title() for t in terms[:3])  # fallback when Groq is not configured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["minilm", "tfidf"], default="minilm")
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    df = db.load_reviews()
    d = df[(df["sentiment"] != "positive") & (df["issue"] != POSITIVE_ISSUE)].reset_index(drop=True)
    print(f"Clustering {len(d)} complaint reviews with {a.embedder} + HDBSCAN ...")
    labels = cluster(embed(d["text_clean"].tolist(), a.embedder), a.min_cluster_size)
    d["cluster"] = labels
    print(f"{labels.max() + 1} clusters, {(labels == -1).sum()} noise reviews")

    names = {}
    for c in sorted(set(labels) - {-1}):
        part = d[d["cluster"] == c]
        samples = part.sample(min(8, len(part)), random_state=42)["text"].tolist()
        names[c] = name_cluster(samples, top_terms(part["text_clean"].tolist()))
        print(f"  cluster {c:>3}  n={len(part):<5} -> {names[c]}")
    if a.dry_run:
        return
    with db.connect() as con:
        for row in d[d["cluster"] >= 0].itertuples():
            issue = names[row.cluster]
            urg = urgency_of(row.rating, row.sentiment, issue, row.emotion, row.red_flag)
            con.execute("UPDATE reviews SET issue=?, issue_source='hdbscan', urgency=? WHERE id=?",
                        (issue, urg, int(row.id)))
    print("empathy.db updated.")


if __name__ == "__main__":
    main()
