import json
import os
import re
import time
from pathlib import Path

import pandas as pd

try:
    from groq import Groq, RateLimitError
except ImportError as exc:
    raise ImportError(
        "The 'groq' package is required to run this script. Install it with: pip install groq"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTER_FILE = BASE_DIR / "outputs" / "issues" / "hdbscan_clustered_reviews.xlsx"
SAMPLE_FILE = BASE_DIR / "outputs" / "issues" / "cluster_representative_samples.csv"
OUTPUT_CSV_FILE = BASE_DIR / "outputs" / "issues" / "cluster_topics_labeled.csv"
OUTPUT_EXCEL_FILE = BASE_DIR / "outputs" / "issues" / "cluster_topics_labeled.xlsx"
FREQUENCY_EXCEL_FILE = BASE_DIR / "outputs" / "issues" / "issue_label_frequency.xlsx"

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def load_cluster_data() -> pd.DataFrame:
    if not CLUSTER_FILE.exists():
        raise FileNotFoundError(
            f"Could not find clustered reviews file: {CLUSTER_FILE}. "
            "Please run scripts/run_hdbscan.py or scripts/cluster_reviews.py first."
        )

    df = pd.read_excel(CLUSTER_FILE)
    if "cluster" not in df.columns:
        raise ValueError(f"Expected a 'cluster' column in {CLUSTER_FILE}")

    if "review_text" not in df.columns and "review" in df.columns:
        df["review_text"] = df["review"]

    return df


def load_cluster_samples(df: pd.DataFrame) -> dict:
    if SAMPLE_FILE.exists():
        sample_df = pd.read_csv(SAMPLE_FILE)
        if "cluster" in sample_df.columns and "review_text" in sample_df.columns:
            return {
                int(cluster_id): group.head(5).to_dict(orient="records")
                for cluster_id, group in sample_df.groupby("cluster")
            }

    samples_by_cluster = {}
    for cluster_id in sorted(set(df["cluster"].dropna().unique()) - {-1}):
        review_rows = (
            df.loc[df["cluster"] == cluster_id, ["restaurant", "rating", "review_text"]]
            .dropna(subset=["review_text"])
            .head(5)
            .to_dict(orient="records")
        )
        samples_by_cluster[int(cluster_id)] = review_rows

    return samples_by_cluster


def parse_retry_seconds(message: str) -> float:
    match = re.search(
        r"try again in\s+(?:(?P<minutes>\d+)m)?\s*(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
        message,
        flags=re.IGNORECASE,
    )
    if not match:
        return 60.0

    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return minutes * 60 + seconds + 2


def normalize_issue_label(label) -> str:
    if pd.isna(label):
        return "Unclassified"

    cleaned = str(label).strip()
    if not cleaned:
        return "Unclassified"

    cleaned = re.sub(r"\s+", " ", cleaned)
    lowered = cleaned.lower()

    if lowered.startswith("none") or "positive" in lowered or "satisfaction" in lowered:
        return "Positive Feedback"
    if "excellent service" in lowered:
        return "Excellent Service"
    if "air conditioning" in lowered or "ac" in lowered:
        return "Air Conditioning Issues"
    if any(token in lowered for token in ["overpriced", "price", "expensive", "cost", "value"]):
        return "Pricing Concerns"
    if any(token in lowered for token in ["service", "staff", "slow", "rude", "attentive", "order"]):
        return "Service Issues"
    if "parse_error" in lowered:
        return "Unclassified"
    return cleaned


def normalize_topic(topic: dict) -> dict:
    normalized_topic = dict(topic)
    if "issue_label" in normalized_topic:
        normalized_topic["issue_label"] = normalize_issue_label(normalized_topic.get("issue_label"))
    if "category" in normalized_topic and pd.isna(normalized_topic.get("category")):
        normalized_topic["category"] = "Other"
    return normalized_topic


def save_topics(topics: list[dict]):
    topics_df = pd.DataFrame([normalize_topic(topic) for topic in topics])
    if "issue_label" in topics_df.columns:
        topics_df["issue_label"] = topics_df["issue_label"].fillna("Unclassified")
    if "category" in topics_df.columns:
        topics_df["category"] = topics_df["category"].fillna("Other")
    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(OUTPUT_CSV_FILE, index=False)
    topics_df.to_excel(OUTPUT_EXCEL_FILE, index=False)
    return topics_df


def load_existing_topics() -> dict[int, dict]:
    if not OUTPUT_CSV_FILE.exists():
        return {}

    try:
        existing_df = pd.read_csv(OUTPUT_CSV_FILE)
    except Exception:
        return {}

    topics_by_cluster: dict[int, dict] = {}
    if "cluster_id" not in existing_df.columns:
        return topics_by_cluster

    for row in existing_df.to_dict(orient="records"):
        try:
            cluster_key = int(row.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        topics_by_cluster[cluster_key] = row

    return topics_by_cluster


def heuristic_label_cluster(cluster_id, reviews, cluster_size):
    if not reviews:
        return {
            "cluster_id": cluster_id,
            "cluster_size": 0,
            "issue_label": "NO_REVIEWS",
            "category": "Other",
            "summary": "No review samples were available for this cluster.",
            "severity": "Low",
            "sample_quote_paraphrase": "No sample available.",
        }

    text_blob = " ".join(
        str(r.get("review_text", "")) for r in reviews if r.get("review_text")
    ).lower()

    if any(word in text_blob for word in ["wait", "crowd", "crowded", "queue", "seating"]):
        issue_label = "Crowd and Wait Time"
        category = "Wait Time/Crowding"
        severity = "High"
        summary = "Customers are frustrated by long waits and crowded conditions that make the dining experience uncomfortable."
        sample = "Customers had to wait a long time and struggled with crowded seating."
    elif any(word in text_blob for word in ["service", "staff", "slow", "rude", "attentive", "order"]):
        issue_label = "Poor Service Quality"
        category = "Service Efficiency"
        severity = "High"
        summary = "Customers report slow, inattentive, or rude service that makes the restaurant experience frustrating."
        sample = "Staff were slow, inattentive, or failed to handle requests properly."
    elif any(word in text_blob for word in ["food", "taste", "flavor", "sambar", "dosa", "roast", "parotta"]):
        issue_label = "Food Quality Issues"
        category = "Food Quality"
        severity = "Medium"
        summary = "Customers raise concerns about inconsistent taste, poor preparation, or disappointing food quality."
        sample = "Food was bland, poorly prepared, or did not meet expectations."
    elif any(word in text_blob for word in ["price", "expensive", "cost", "value"]):
        issue_label = "Pricing Concerns"
        category = "Pricing/Value"
        severity = "Medium"
        summary = "Customers feel the restaurant is overpriced or not offering enough value for the money."
        sample = "The food or experience felt too expensive for the value provided."
    elif any(word in text_blob for word in ["hygiene", "dirty", "clean", "washroom", "plate", "utensil"]):
        issue_label = "Poor Hygiene Standards"
        category = "Hygiene"
        severity = "High"
        summary = "Customers report unhygienic conditions, dirty utensils, or poor cleanliness in the restaurant."
        sample = "The restaurant felt dirty and cleanliness standards were poor."
    elif any(word in text_blob for word in ["ac", "air", "cool", "ventilation"]):
        issue_label = "Poor Air Conditioning"
        category = "Ambience"
        severity = "Medium"
        summary = "Customers complain about uncomfortable temperatures due to weak or missing cooling."
        sample = "The restaurant felt too hot and uncomfortable because of poor cooling."
    else:
        issue_label = "General Feedback"
        category = "Other"
        severity = "Low"
        summary = "The reviews are mostly positive or do not point to one dominant complaint."
        sample = "Customers describe a generally positive or mixed experience."

    return {
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "issue_label": issue_label,
        "category": category,
        "summary": summary,
        "severity": severity,
        "sample_quote_paraphrase": sample,
    }


def label_cluster(cluster_id, reviews, cluster_size, model="llama-3.3-70b-versatile"):
    if not reviews:
        return {
            "cluster_id": cluster_id,
            "cluster_size": 0,
            "issue_label": "NO_REVIEWS",
            "category": "Other",
            "summary": "No review samples were available for this cluster.",
            "severity": "Low",
            "sample_quote_paraphrase": "No sample available.",
        }

    if client is None:
        existing_topics = load_existing_topics()
        if cluster_id in existing_topics:
            topic = existing_topics[cluster_id].copy()
            topic["cluster_id"] = cluster_id
            topic["cluster_size"] = cluster_size
            return topic
        return heuristic_label_cluster(cluster_id, reviews, cluster_size)

    review_block = "\n".join(
        f"{i + 1}. [{r.get('restaurant', 'unknown')}, rating={r.get('rating', 'n/a')}] {r.get('review_text', '')}"
        for i, r in enumerate(reviews)
    )

    system_prompt = """You are a restaurant customer-experience analyst.
You will be given a set of customer reviews that a clustering algorithm has grouped
together because they are semantically similar. Your job is to identify the SINGLE
dominant complaint or theme uniting these reviews.

Return ONLY valid JSON, no preamble, no markdown fences, in this exact schema:
{
  "issue_label": "short 3-6 word business-facing label, e.g. 'Slow Table Service'",
  "category": "one of: Food Quality, Service Efficiency, Ambience, Hygiene, Pricing/Value, Staff Behaviour, Wait Time/Crowding, Order Accuracy, Other",
  "summary": "2-3 sentence plain-language summary of what customers are complaining about",
  "severity": "High, Medium, or Low - based on how strongly worded/frequent the frustration seems",
  "sample_quote_paraphrase": "a short paraphrased (not verbatim) description of a typical complaint in this cluster"
}"""

    user_prompt = f"Reviews in this cluster:\n\n{review_block}"

    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except RateLimitError as exc:
            wait_seconds = parse_retry_seconds(str(exc))
            print(
                f"Rate limit reached while labeling cluster {cluster_id}. "
                f"Retrying in {wait_seconds:.0f} seconds (attempt {attempt}/5)..."
            )
            if attempt == 5:
                raise
            time.sleep(wait_seconds)
    else:
        raise RuntimeError(f"Failed to label cluster {cluster_id} after repeated rate-limit retries.")

    raw_text = response.choices[0].message.content.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"issue_label": "PARSE_ERROR", "raw": raw_text}

    result["cluster_id"] = cluster_id
    result["cluster_size"] = cluster_size
    return normalize_topic(result)


def main():
    df = load_cluster_data()
    cluster_samples = load_cluster_samples(df)

    cluster_topics = []
    completed_cluster_ids = set()
    if OUTPUT_CSV_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_CSV_FILE)
        if "cluster_id" in existing_df.columns:
            cluster_topics = existing_df.to_dict(orient="records")
            completed_cluster_ids = set(existing_df["cluster_id"].astype(int).tolist())

    for cluster_id, samples in cluster_samples.items():
        if cluster_id in completed_cluster_ids:
            continue

        cluster_size = int((df["cluster"] == cluster_id).sum())
        print(f"Labeling cluster {cluster_id} ({len(samples)} sampled reviews, cluster_size={cluster_size})...")
        topic = label_cluster(cluster_id, samples, cluster_size)
        cluster_topics.append(topic)
        save_topics(cluster_topics)

    if not cluster_topics:
        print("No clusters were available for labeling.")
        return

    topics_df = save_topics(cluster_topics)
    print(f"Saved labeled cluster topics to: {OUTPUT_CSV_FILE}")
    print(f"Saved labeled cluster topics to Excel: {OUTPUT_EXCEL_FILE}")
    if "issue_label" in topics_df.columns:
        counts = topics_df["issue_label"].fillna("Unclassified").astype(str).value_counts().reset_index()
        counts.columns = ["issue_label", "frequency"]
        counts.to_excel(FREQUENCY_EXCEL_FILE, index=False)
        print("\nNormalized issue-label frequency:")
        print(counts.to_string(index=False))
    print(pd.DataFrame(cluster_topics))


if __name__ == "__main__":
    main()

