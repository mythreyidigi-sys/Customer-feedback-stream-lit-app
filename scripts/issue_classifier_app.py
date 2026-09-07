"""Standalone Streamlit interface for the trained issue classifier."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "scripts" / "issue_classifier.joblib"

st.set_page_config(page_title="Issue Classifier", layout="wide")
st.title("Issue Classifier")
st.caption("Classify customer feedback into the most likely restaurant issue category.")


@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["vectorizer"]


model, vectorizer = load_classifier()
if model is None or vectorizer is None:
    st.error("The trained issue classifier is not available.")
    st.stop()

with st.form("classify_review"):
    review_text = st.text_area("Customer review", height=180)
    submitted = st.form_submit_button("Classify review", type="primary")

if submitted:
    if not review_text.strip():
        st.warning("Enter a customer review before classifying it.")
    else:
        probabilities = model.predict_proba(vectorizer.transform([review_text]))[0]
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = ranked_indices[0]

        with st.container(border=True):
            st.metric("Predicted issue", model.classes_[best_index])
            st.metric("Confidence", f"{probabilities[best_index]:.1%}")

        top_indices = ranked_indices[:3]
        st.subheader("Top category matches")
        st.dataframe(
            pd.DataFrame(
                {
                    "Issue category": model.classes_[top_indices],
                    "Confidence": probabilities[top_indices],
                }
            ),
            column_config={"Confidence": st.column_config.NumberColumn(format="%.1%%")},
            hide_index=True,
            width="stretch",
        )