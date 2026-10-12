"""Component 2 - SHAP explanations for the trained XGBoost issue-priority model.

Run AFTER issue_priority_model.py:  python shap_explain.py
SHAP explains the already-trained model; it is not a separate predictive model. Contributions are in
log-odds of the "sentiment declines next month" prediction and describe the model, not causal effects.

Outputs (outputs/xgb_priority/shap/): global_importance.csv, shap_summary.png, local_explanations.json,
local_<k>.png, stability.csv, stability_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import issue_priority_model as ipm  # noqa: E402

SRC = ipm.OUT_DIR
OUT = SRC / "shap"
N_LOCAL = 3
N_STABILITY = 20
TOP_K = 5


def pretty(name: str) -> str:
    name = name.split("__", 1)[1] if name.startswith(("num__", "cat__")) else name
    return (name.replace("share__", "Share of: ").replace("restaurant_", "Restaurant: ")
            .replace("source_", "Source: ").replace("sent_mean", "Mean sentiment this month")
            .replace("neg_share", "Share of negative reviews").replace("n_reviews", "Review volume")
            .replace("complaint_share", "Total complaint share"))


def shap_values(model, X: np.ndarray) -> np.ndarray:
    return shap.TreeExplainer(model).shap_values(X)


def explain_sentence(row: pd.Series, contrib: pd.Series, prob: float, top: int = 3) -> str:
    """Plain-language summary of the strongest drivers for one unit-month."""
    parts = []
    for f, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top].items():
        parts.append(f"{f} ({'raises' if v > 0 else 'lowers'} the risk)")
    return (f"For {row['restaurant']} on {row['source']} in {row['month']}, the model puts the chance that "
            f"sentiment declines next month at {prob:.0%}. The biggest drivers were: " + "; ".join(parts) +
            ". These are patterns the model learned, not proven causes.")


def stability(model_X: np.ndarray, y: np.ndarray, names: list[str], base_imp: pd.Series,
              ipm_num, ipm_cat, panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Refit on bootstrap resamples (re-fitting the preprocessor too) and compare mean|SHAP| rankings
    with the reference model, always evaluated on the same reference rows."""
    rng = np.random.default_rng(ipm.SEED)
    rows, tries = [], 0
    X_df = panel[feats]
    while len(rows) < N_STABILITY and tries < N_STABILITY * 5:
        tries += 1
        idx = rng.integers(0, len(panel), len(panel))
        if len(set(y[idx])) < 2:
            continue
        pre_b = ipm.make_preprocessor(ipm_num, ipm_cat).fit(X_df.iloc[idx])
        m_b = ipm.make_xgb(float((y[idx] == 0).sum() / max((y[idx] == 1).sum(), 1)))
        m_b.fit(pre_b.transform(X_df.iloc[idx]), y[idx])
        sv = shap_values(m_b, pre_b.transform(X_df))
        imp = pd.Series(np.abs(sv).mean(axis=0), index=[pretty(n) for n in pre_b.get_feature_names_out()])
        imp = imp.reindex(base_imp.index).fillna(0)
        rho = spearmanr(base_imp.values, imp.values)[0]
        overlap = len(set(base_imp.nlargest(TOP_K).index) & set(imp.nlargest(TOP_K).index)) / TOP_K
        rows.append({"resample": len(rows), "spearman_rank_corr": rho, f"top{TOP_K}_overlap": overlap,
                     **{f"imp::{k}": v for k, v in imp.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(SRC / "panel.csv")
    spec = json.loads((SRC / "feature_list.json").read_text())
    num, cat = spec["numeric"], spec["categorical"]
    feats = num + cat
    pre = joblib.load(SRC / "preprocessing.joblib")
    model = XGBClassifier()
    model.load_model(str(SRC / "model.json"))

    X = pre.transform(panel[feats])
    names = [pretty(n) for n in pre.get_feature_names_out()]
    sv = shap_values(model, X)
    prob = model.predict_proba(X)[:, 1]

    imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    direction = pd.Series([np.corrcoef(X[:, j], sv[:, j])[0, 1] if X[:, j].std() > 0 and sv[:, j].std() > 0
                           else 0.0 for j in range(X.shape[1])], index=names)
    pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values,
                  "value_vs_risk_corr": direction.reindex(imp.index).values}).to_csv(
        OUT / "global_importance.csv", index=False)

    shap.summary_plot(sv, X, feature_names=names, max_display=12, show=False)
    plt.title("SHAP summary: drivers of predicted next-month sentiment decline (log-odds)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=130)
    plt.close()

    # local explanations for the highest-risk predictions
    local = []
    for k, i in enumerate(np.argsort(-prob)[:N_LOCAL]):
        contrib = pd.Series(sv[i], index=names)
        exp = shap.Explanation(values=sv[i], base_values=float(np.ravel(shap.TreeExplainer(model).expected_value)[0]),
                               data=X[i], feature_names=names)
        shap.plots.waterfall(exp, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(OUT / f"local_{k}.png", dpi=130)
        plt.close()
        row = panel.iloc[i]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6]
        local.append({"id": k, "restaurant": row["restaurant"], "source": row["source"], "month": row["month"],
                      "predicted_decline_probability": float(prob[i]), "actual_decline": int(row["target_decline"]),
                      "top_contributions": [{"feature": f, "shap": float(v)} for f, v in top.items()],
                      "explanation": explain_sentence(row, contrib, float(prob[i])), "plot": f"local_{k}.png"})
    (OUT / "local_explanations.json").write_text(json.dumps(local, indent=2))

    stab = stability(X, panel["target_decline"].values, names, imp, num, cat, panel, feats)
    stab.to_csv(OUT / "stability.csv", index=False)
    imp_cols = [c for c in stab.columns if c.startswith("imp::")]
    imp_sd = stab[imp_cols].std().rename(lambda c: c[5:])
    summary = {"resamples": len(stab), "mean_spearman": float(stab["spearman_rank_corr"].mean()),
               "min_spearman": float(stab["spearman_rank_corr"].min()),
               f"mean_top{TOP_K}_overlap": float(stab[f"top{TOP_K}_overlap"].mean()),
               "most_stable_top_features": [f for f in imp.index[:TOP_K]],
               "importance_sd_top": {f: float(imp_sd[f]) for f in imp.index[:TOP_K]},
               "stable": bool(stab["spearman_rank_corr"].mean() >= 0.7 and stab[f"top{TOP_K}_overlap"].mean() >= 0.6)}
    (OUT / "stability_summary.json").write_text(json.dumps(summary, indent=2))
    print(imp.head(10).round(3).to_string(), "\n")
    print(json.dumps(summary, indent=2), "\n")
    for e in local:
        print(e["explanation"])


if __name__ == "__main__":
    main()
