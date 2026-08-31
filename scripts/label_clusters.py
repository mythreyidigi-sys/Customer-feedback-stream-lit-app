import pandas as pd

# Load clustered reviews
df = pd.read_excel("outputs/issues/clustered_reviews.xlsx")

print("Total Reviews:", len(df))

# Get sample reviews from each cluster
cluster_samples = []

for cluster in sorted(df["cluster"].unique()):

    reviews = df[df["cluster"] == cluster]["review"].dropna()

    sample_reviews = reviews.head(5).tolist()

    cluster_samples.append({
        "cluster": cluster,
        "sample_reviews": "\n\n".join(sample_reviews)
    })

sample_df = pd.DataFrame(cluster_samples)

sample_df.to_excel(
    "outputs/issues/cluster_samples.xlsx",
    index=False
)

print("Saved: outputs/issues/cluster_samples.xlsx")