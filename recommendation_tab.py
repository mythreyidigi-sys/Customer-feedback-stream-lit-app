"""
recommendation_tab.py - add to orm_app.py:

    from recommendation_tab import render_recommendation_tab
    ...
    with tab_reco:            # new tab named "Recommendations"
        render_recommendation_tab(df)
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from recommendation_engine import (PLAYBOOK, recommend, find_hotspots,
                                   tailor_with_llm, run_simulation, canonical_issue)


def render_recommendation_tab(df: pd.DataFrame, chain_col="restaurant",
                              issue_col="issue_cluster", date_col=None):
    st.header("Recommendation Engine")

    if df is None or df.empty:
        st.info("No reviews match the current filters.")
        return
    required_columns = {chain_col, issue_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        st.warning("Recommendation analysis needs these columns: " + ", ".join(missing_columns))
        return
    df = df.dropna(subset=[chain_col, issue_col]).copy()
    if df.empty:
        st.info("No classified reviews are available for recommendations yet.")
        return

    # --- A. Hotspots ---------------------------------------------------
    st.subheader("1. Where to act first: issue hotspots")
    hs = find_hotspots(df, chain_col, issue_col, date_col)
    if hs.empty:
        st.info("Not enough issue data to identify hotspots yet.")
    else:
        st.caption(f"Method: {hs['method'].iloc[0]}. Lift > 1.3 means the chain has at least "
                   f"30% more of this issue than the six-chain average.")
        st.dataframe(hs.drop(columns="method"), use_container_width=True)

    # --- B. Single review recommendation --------------------------------
    st.subheader("2. Recommendation for a specific review")
    c1, c2, c3 = st.columns(3)
    issue = c1.selectbox("Issue category", list(PLAYBOOK.keys()))
    emotion = c2.selectbox("Emotion", ["frustration", "betrayal", "disappointment",
                                       "neutral", "delight"])
    urgency = c3.slider("Urgency score", 0, 10, 5)
    chain = st.selectbox("Restaurant", sorted(df[chain_col].dropna().unique()), key="recommendation_chain")
    text = st.text_area("Review text (optional, used for AI tailoring)")

    rec = recommend(issue, emotion, urgency)
    st.markdown(f"**Severity:** {rec['severity'].upper()}"
                + ("  ⚠️ escalate to branch manager" if rec["escalate"] else ""))
    st.table(pd.DataFrame(rec["actions"])[["action", "owner", "timeframe", "kpi"]])
    if text and st.button("Tailor with AI (Groq)"):
        st.write(tailor_with_llm(text, chain, rec))

    # --- C. Bandit proof of concept ------------------------------------
    st.subheader("3. Learning which actions work: contextual bandit (simulated)")
    st.warning("Proof of concept on a SIMULATED environment. Action success rates are "
               "assumptions, not observed outcomes. With real intervention logs, the same "
               "algorithm would learn from actual results.")
    rounds = st.slider("Simulated weeks x complaints", 500, 10000, 5000, step=500)
    if st.button("Run simulation"):
        res = run_simulation(rounds)
        cum = res["rewards"].expanding().mean().iloc[50:]
        st.plotly_chart(px.line(cum, labels={"index": "Round", "value": "Avg success rate",
                                             "variable": "Policy"},
                                title="Average success rate by policy"),
                use_container_width=True)
        st.dataframe(res["policy"], use_container_width=True)
