import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("outputs/issues/top5_issues.xlsx")

plt.figure(figsize=(10,6))
plt.bar(df["Issue"], df["Frequency"])

plt.xticks(rotation=30, ha="right")
plt.ylabel("Frequency")
plt.title("Top 5 Customer Issues Across Restaurant Chains")

plt.tight_layout()

plt.savefig(
    "outputs/charts/issue_bar_chart.png"
)

print("Chart Saved!")