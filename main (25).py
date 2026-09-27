"""Empathy Engine — Streamlit decision-support app (single entry point).

Dissertation: Leveraging Customer Experience Analytics for Service Quality Improvement in
High-Volume Restaurants (N.R. Mythreyi · 2024MB22535 · BITS Pilani WILP, MBA Digital Business)

Run:  streamlit run main.py

Pipeline (final dataset: 6 chains x 3 platforms = 18 combinations):
    Google / Zomato / TripAdvisor (CSV/Excel, scraped with Selenium + Playwright)
      -> Import pipeline (clean names, branches, dates, dedupe)
      -> Sentence-Transformer (all-MiniLM-L6-v2) embeddings
      -> HDBSCAN clustering (no preset k; outliers flagged as noise)
      -> Groq LLM cluster labels (issue), plus emotion / urgency classifier
      -> SQLite (empathy.db: reviews, actions, reports, runs)
      -> Streamlit app:
           1. Search            data coverage by chain and platform
           2. Overview          KPIs, top issues, priority decision panel
           3. Early warning     Module 6 - reputation early-warning & resolution
           4. Location & ambience   Module 5 - branch-level location & ambience intelligence
           5. Campaigns         Digital marketing / 360-degree campaigns tied to clusters
           6. Monthly report    PDF summary for managers

Setup:
    pip install -r requirements.txt
    python -m engine.importer --reset --scraped-on YYYY-MM-DD
    set GROQ_API_KEY=gsk_...                                # optional AI replies
    streamlit run main.py

This file needs the `engine/` package. It looks for it next to main.py first,
then in empathy_engine/ and scripts_new792026/empathy_engine (5)/empathy_engine/.
"""
import difflib
import sys
from datetime import date, datetime
from pathlib import Path

# ------------------------------------------------------------------ locate the engine package
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE,
    _HERE / "empathy_engine",
    _HERE / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine",
]
APP_DIR = next((p for p in _CANDIDATES if (p / "engine").is_dir()), _HERE)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import analytics as an
from engine import db
from engine.config import (HIGH_THRESHOLD, POSITIVE_ISSUE, RAW_DIR, SOURCES, TEAM, URGENT_THRESHOLD)
from engine.importer import import_file, source_from_filename
from engine.playbook import deadline_text, recommend
from engine.replies import draft_reply, groq_available, template_reply
from engine.report import build_report, save_report

st.set_page_config(page_title="Empathy Engine", page_icon="💬", layout="wide")
db.init_db()

GREEN, RED, GREY, AMBER, ACCENT = "#1D9E75", "#D85A30", "#B4B2A9", "#BA7517", "#0F6E56"
SENT_COLORS = {"negative": RED, "neutral": GREY, "positive": GREEN}
PAGES = ["🔍 1. Search", "📊 2. Overview", "🚨 3. Early warning", "📍 4. Location & ambience",
         "📣 5. Campaigns", "📄 6. Monthly report"]

# ------------------------------------------------------------------ project reference figures
# Final dataset as reported in the dissertation (used only as a reference caption; live
# numbers in the app always come from the database).
PROJECT_REF = {"raw": 6609, "clean": 6159, "classified": 5909, "categories": 10,
               "chains": 6, "platforms": 3, "branches": 108}
EXPECTED_CHAINS = ["A2B", "Sangeetha", "Saravana Bhavan", "Sree Annapoorna",
                   "Namma Veedu Vasantha Bhavan", "Geetham"]
NOISE_LABELS = {"noise", "unclassified", "uncategorized", "uncategorised", "outlier", "-1", "other", ""}

# ------------------------------------------------------------------ Module 5 aspects
ASPECTS = {
    "Parking": r"\bpark(ing)?\b|two[- ]?wheeler|valet",
    "Seating & crowding": r"\bseat(s|ing)?\b|\btable(s)?\b|crowd(ed)?|\brush\b|\bqueue\b|no place|space",
    "Ambience & noise": r"ambi[ae]nce|atmosphere|\bnois(e|y)\b|\bdecor\b|lighting|music|\bac\b|air[- ]?condition|\bhot inside\b",
    "Cleanliness": r"clean|dirty|hygien|unhygien|washroom|restroom|toilet|\bflies\b|cockroach|smell",
    "Location & access": r"locat(ion|ed)|easy to find|hard to find|\baccess\b|near (the )?(bus|metro|station)|main road|landmark",
}

# ------------------------------------------------------------------ 360-degree campaigns
# Each campaign is tied to an HDBSCAN issue cluster. Matching is keyword-based on the cluster
# label so it keeps working when the Groq labels change slightly after a re-run.
# Order matters: the first campaign whose keywords match wins.
CAMPAIGNS = [
    dict(id=5, name="Clean Kitchen, Clear Conscience", theme="Poor experience & food hygiene",
         keys=("hygien", "clean", "dirty", "food safety", "poor experience"),
         headline="Our kitchen, your peace of mind — see our hygiene standards for yourself.",
         channels=["In-store signage", "Owned/PR", "Review-platform response"],
         kpi="Zero tolerance: any week-over-week spike pauses the public campaign",
         mode="risk"),
    dict(id=2, name="More to Choose, Faster to Get", theme="Food variety & fast service",
         keys=("variety", "menu", "choice", "option"),
         headline="More dishes. Less waiting. Now on the menu.",
         channels=["Paid social", "Zomato/Swiggy in-app", "In-store signage", "Email/SMS"],
         kpi="Trial rate of promoted dishes; order-to-serve time vs baseline",
         mode="launch"),
    dict(id=4, name="On Time, Every Time", theme="Slow service & staff negligence",
         keys=("slow", "wait", "delay", "negligen", "late", "queue"),
         headline="We cut our average wait time. Come time us.",
         channels=["Paid social", "Google Ads", "Zomato/Swiggy in-app", "Influencer/UGC"],
         kpi="Table-turnaround time; 'wait/slow/negligence' mentions in new reviews",
         mode="gated"),
    dict(id=1, name="Full Plate Promise", theme="Food quantity & value for money",
         keys=("quantity", "value", "portion", "price", "money", "cost", "expensive"),
         headline="Full plate. Fair price. No compromises on either.",
         channels=["Paid social", "In-store signage", "Influencer/UGC"],
         kpi="Cut this cluster's share of new reviews by 15–20% within one quarter",
         mode="launch"),
    dict(id=3, name="Service, Reimagined", theme="Service quality",
         keys=("service", "staff", "rude", "behav", "courte", "attitude", "waiter"),
         headline="Every order matters. Every guest matters. That's our new standard.",
         channels=["Paid social", "In-store signage", "Owned/PR", "Review-platform response"],
         kpi="Lower Service Quality share; higher star rating on 'staff/service' reviews",
         mode="launch"),
]
CHANNELS = ["Paid social", "Google Ads", "Zomato/Swiggy in-app", "In-store signage",
            "Influencer/UGC", "Owned/PR", "Email/SMS", "Review-platform response"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.pill {display:inline-block; font-size:12px; padding:2px 10px; border-radius:999px; margin:0 4px 4px 0;
       background:rgba(128,128,128,.15);}
.p-urgent {background:#FCEBEB; color:#A32D2D;} .p-high {background:#FAEEDA; color:#854F0B;}
.p-ok {background:#E1F5EE; color:#0F6E56;} .p-muted {opacity:.65;}
.quote {font-style:italic; font-size:15px; margin:6px 0 10px;}
.small {font-size:13px; opacity:.75;}
</style>""", unsafe_allow_html=True)


def pill(text, cls=""):
    return f'<span class="pill {cls}">{text}</span>'


AUTO_DETECT_SOURCE = "Auto-detect from filename"


def resolve_upload_source(file_name, selected_source):
    if selected_source != AUTO_DETECT_SOURCE:
        return selected_source
    source = source_from_filename(file_name)
    if source:
        return source
    raise ValueError(f"Can't detect the source from '{file_name}'. Choose Google, Zomato, or TripAdvisor manually.")


# ------------------------------------------------------------------ helpers (pure pandas)
def is_noise(issue: pd.Series) -> pd.Series:
    """HDBSCAN outliers (label -1) or rows without a usable cluster label."""
    s = issue.astype("string").str.strip().str.lower()
    return s.isna() | s.isin(NOISE_LABELS)


def classified(df: pd.DataFrame) -> pd.DataFrame:
    return df[~is_noise(df["issue"])]


def complaint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Classified, non-positive reviews: the complaint clusters."""
    d = classified(df)
    return d[d["issue"] != POSITIVE_ISSUE]


def is_negative(df: pd.DataFrame) -> pd.Series:
    if "sentiment" in df.columns:
        return df["sentiment"].eq("negative") | df["rating"].le(2)
    return df["rating"].le(2)


def match_campaign(issue):
    label = str(issue).lower()
    for c in CAMPAIGNS:
        if any(k in label for k in c["keys"]):
            return c["id"]
    return None


def weekly_counts(df: pd.DataFrame, as_of, weeks=5) -> list:
    """Counts for the last `weeks` 7-day blocks ending at as_of (oldest first)."""
    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    out = []
    for w in range(weeks, 0, -1):
        lo, hi = end - pd.Timedelta(days=7 * w), end - pd.Timedelta(days=7 * (w - 1))
        out.append(int(((df["review_date"] >= lo) & (df["review_date"] < hi)).sum()))
    return out


def aspect_flags(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("").str.lower()
    return pd.DataFrame({a: text.str.contains(rx, regex=True) for a, rx in ASPECTS.items()}, index=df.index)


# ------------------------------------------------------------------ data + sidebar filters
df_all = db.load_reviews()

if df_all.empty:
    st.title("💬 Empathy Engine")
    st.warning("No reviews in the database yet.")
    st.markdown("""
**Load your data in one of two ways**

1. Put your files in `data/raw/google/`, `data/raw/zomato/`, `data/raw/tripadvisor/` and run
   `python -m engine.importer --scraped-on YYYY-MM-DD`
2. Or upload them below.

Files that already carry an HDBSCAN issue / cluster label column keep that label.
""")
    source = st.selectbox("Review platform", [AUTO_DETECT_SOURCE, *SOURCES])
    ups = st.file_uploader("Review files (CSV or Excel)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    ref = st.date_input("Date the files were scraped (for '2 months ago' dates)", date.today())
    if ups and st.button("Import", type="primary"):
        for u in ups:
            try:
                upload_source = resolve_upload_source(u.name, source)
            except ValueError as exc:
                st.error(str(exc))
                continue
            path = RAW_DIR / upload_source.lower() / u.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(u.getbuffer())
            st.write(import_file(path, upload_source, ref, force_source=True))
        st.rerun()
    st.stop()

restaurants = sorted(df_all["restaurant"].unique())
data_end = df_all["review_date"].max().date()
data_start = df_all["review_date"].min().date()
ss = st.session_state
ss.setdefault("restaurant", "Geetham" if "Geetham" in restaurants else restaurants[0])
ss.setdefault("page", PAGES[0])
if ss.page not in PAGES:          # session from an older version of the app
    ss.page = PAGES[0]

with st.sidebar:
    st.markdown("### 💬 Empathy Engine")
    st.selectbox("Restaurant", restaurants, key="restaurant")
    rest_df = df_all[df_all["restaurant"] == ss.restaurant]
    branches = st.multiselect("Branches", sorted(rest_df["branch"].unique()), placeholder="All branches")
    sources = st.multiselect("Sources", SOURCES, default=SOURCES)
    days = st.select_slider("Window", options=[7, 14, 30, 60, 90, 180, 365], value=90,
                            format_func=lambda d: f"Last {d} days")
    as_of = st.date_input("As of", data_end, min_value=data_start, max_value=data_end)
    compare = st.multiselect("Compare with", [r for r in restaurants if r != ss.restaurant],
                             default=[r for r in ("A2B", "Sangeetha") if r in restaurants and r != ss.restaurant])
    st.divider()
    st.caption(f"Data: {len(df_all):,} reviews · {df_all['restaurant'].nunique()} chains · "
               f"{df_all['branch'].nunique()} branches · {df_all['source'].nunique()} platforms")
    st.caption(f"{data_start:%d %b %Y} – {data_end:%d %b %Y}")
    st.caption("Clusters: Sentence-Transformer + HDBSCAN · labels by Groq LLM")
    st.caption("Replies: " + ("Groq LLM ✅" if groq_available() else "templates (set GROQ_API_KEY for AI replies)"))

(cur_start, cur_end), (prev_start, prev_end) = an.windows(as_of, days)
scope = an.filter_reviews(df_all, ss.restaurant, branches or None, sources)
cur = an.filter_reviews(scope, start=cur_start, end=cur_end)
prev = an.filter_reviews(scope, start=prev_start, end=prev_end)
upto = an.filter_reviews(scope, end=as_of)
spikes = an.spikes(upto, as_of)
esc_open, esc_overdue = an.open_escalations(upto, as_of)

# ------------------------------------------------------------------ header + navigation
h1, h2 = st.columns([3, 2])
h1.markdown("## 💬 Empathy Engine")
h2.markdown(f"<div style='text-align:right;padding-top:14px'>🏪 <b>{ss.restaurant}</b> · "
            f"{', '.join(branches) if branches else 'all branches'} &nbsp; 🔔 "
            f"{pill(str(len(esc_open)), 'p-urgent')}</div>", unsafe_allow_html=True)
page = st.radio("Screen", PAGES, key="page", horizontal=True, label_visibility="collapsed")
st.divider()


def go_overview():
    ss.page = PAGES[1]


def analyse_callback():
    q = ss.get("search_q", "").strip().lower()
    if not q:
        return
    names = {r.lower(): r for r in restaurants}
    hit = next((names[n] for n in names if q in n), None)
    if not hit:
        close = difflib.get_close_matches(q, list(names), n=1, cutoff=0.3)
        hit = names[close[0]] if close else None
    if hit:
        ss.restaurant = hit
        ss.page = PAGES[1]
        ss.search_msg = None
    else:
        ss.search_msg = f"No restaurant matching '{q}'. Known: {', '.join(restaurants)}"


# ================================================================== 1. SEARCH
if page == PAGES[0]:
    with st.container(border=True):
        st.markdown("<h3 style='text-align:center'>What are customers saying about your restaurant?</h3>"
                    "<p class='small' style='text-align:center'>Type a restaurant name. We'll gather its reviews "
                    "from every source and show what to fix first.</p>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            c1, c2 = st.columns([4, 1])
            c1.text_input("Restaurant", value=ss.restaurant, key="search_q", label_visibility="collapsed",
                          placeholder="e.g. Geetham, A2B, Sangeetha")
            c2.button("Analyse", type="primary", on_click=analyse_callback, width="stretch")
            if ss.get("search_msg"):
                st.warning(ss.search_msg)
        source_counts = rest_df["source"].value_counts()
        pills = "".join(pill(f"✓ {s} · {source_counts.get(s, 0)} total", "p-ok") if source_counts.get(s, 0)
                        else pill(f"{s} · no data", "p-muted") for s in SOURCES)
        st.markdown(f"<div style='text-align:center'><p class='small'>Sources</p>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='small' style='text-align:center'>📅 Last {days} days · {cur_start:%d %b} – "
                    f"{cur_end:%d %b %Y} &nbsp;&nbsp; ⚖️ Compare with: {', '.join(compare) or '—'}</p>",
                    unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"**Reviews collected by source** · {ss.restaurant} · all dates")
        mx = max(1, source_counts.max() if len(source_counts) else 1)
        for s in SOURCES:
            a, b, c = st.columns([1.2, 6, 0.8])
            a.write(s)
            b.progress(int(source_counts.get(s, 0) / mx * 100))
            c.write(f"**{source_counts.get(s, 0)}**")
        st.button("See the overview →", on_click=go_overview)

    with st.container(border=True):
        st.markdown("**Pipeline & dataset** · live from the database")
        n_cls = int((~is_noise(df_all["issue"])).sum())
        n_cat = classified(df_all)["issue"].nunique()
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Reviews", f"{len(df_all):,}", border=True)
        p2.metric("Classified", f"{n_cls:,}", f"{n_cls / max(1, len(df_all)) * 100:.0f}% of reviews",
                  delta_color="off", border=True)
        p3.metric("Issue categories", n_cat, border=True)
        p4.metric("Chains × platforms", f"{df_all.groupby(['restaurant', 'source']).ngroups}/18", border=True)
        p5.metric("Branches", df_all["branch"].nunique(), border=True)
        st.caption(f"Dissertation reference: {PROJECT_REF['raw']:,} raw → {PROJECT_REF['clean']:,} cleaned → "
                   f"{PROJECT_REF['classified']:,} classified into {PROJECT_REF['categories']} HDBSCAN categories "
                   f"(rest flagged as noise) · {PROJECT_REF['chains']} chains · {PROJECT_REF['platforms']} platforms · "
                   f"{PROJECT_REF['branches']} branches.")
        missing = [c for c in EXPECTED_CHAINS if c not in restaurants]
        if missing:
            st.warning("Chains not in the database yet: " + ", ".join(missing))

    with st.expander("📥 Add or update review files (Google / Zomato / TripAdvisor)"):
        c1, c2 = st.columns(2)
        src = c1.selectbox("Review platform", [AUTO_DETECT_SOURCE, *SOURCES])
        ref = c2.date_input("Date the files were scraped", date.today(),
                            help="Used to convert Google's '3 weeks ago' style dates into real dates.")
        ups = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        if ups and st.button("Import files", type="primary"):
            for u in ups:
                try:
                    upload_source = resolve_upload_source(u.name, src)
                except ValueError as exc:
                    st.error(str(exc))
                    continue
                path = RAW_DIR / upload_source.lower() / u.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(u.getbuffer())
                r = import_file(path, upload_source, ref, force_source=True)
                (st.error if r["message"].startswith("ERROR") else st.success)(
                    f"{u.name} ({upload_source}): {r['added']} new of {r['read']} rows. {r['message']}")
            st.button("Refresh")
        st.caption("Duplicates are skipped automatically, so re-uploading a file is safe. "
                   "If your file has an issue/cluster label column (from HDBSCAN), it is kept.")
        runs = db.load_runs()
        if len(runs):
            st.dataframe(runs[["started_at", "source", "file_name", "rows_read", "rows_added", "message"]],
                         hide_index=True, width="stretch")

    with st.expander("🗂️ Data coverage — all 6 chains × 3 platforms"):
        cov = df_all.pivot_table(index="restaurant", columns="source", values="id", aggfunc="count", fill_value=0)
        cov = cov.reindex(columns=SOURCES, fill_value=0)
        cov["Total"] = cov.sum(axis=1)
        cov["Branches"] = df_all.groupby("restaurant")["branch"].nunique()
        cov["From"] = df_all.groupby("restaurant")["review_date"].min().dt.strftime("%d %b %Y")
        cov["To"] = df_all.groupby("restaurant")["review_date"].max().dt.strftime("%d %b %Y")
        st.dataframe(cov, width="stretch")

# ================================================================== 2. OVERVIEW
elif page == PAGES[1]:
    if cur.empty:
        st.info("No reviews in this window. Widen the window or change filters in the sidebar.")
        st.stop()
    for s in spikes.head(3).itertuples():
        st.error(f"**Spike this week:** {s.issue} at **{s.branch}** — {int(s.this_week)} in the last "
                 f"7 days, {s.ratio:.1f}× the 4-week average.", icon="⚠️")

    k = an.kpis(cur, prev, as_of)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Reviews, {days} days", k["reviews"],
              None if k["reviews_change"] is None else f"{k['reviews_change']:+.0f}% vs previous", border=True)
    m2.metric("Reputation score /100", k["score"],
              None if k["score_delta"] is None else f"{k['score_delta']:+.1f} pts", border=True)
    m3.metric("Negative share", f"{k['neg_share']:.0f}%",
              None if k["neg_delta"] is None else f"{k['neg_delta']:+.1f} pts", delta_color="inverse", border=True)
    esc_c, over_c = an.open_escalations(cur, as_of)
    m4.metric("Open escalations", len(esc_c), f"{len(over_c)} overdue" if len(over_c) else "none overdue",
              delta_color="inverse" if len(over_c) else "off", border=True)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Weekly sentiment**")
        ws = an.weekly_sentiment(cur)
        fig = px.bar(ws, x="week", y="reviews", color="sentiment", color_discrete_map=SENT_COLORS,
                     category_orders={"sentiment": ["negative", "neutral", "positive"]})
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend_title=None, xaxis_title=None,
                          legend=dict(orientation="h", y=-0.2), bargap=0.3)
        st.plotly_chart(fig, width="stretch")
    with c2.container(border=True):
        st.markdown("**Reputation score by month** (6 months)")
        mt = an.monthly_trend(upto)
        fig = go.Figure(go.Scatter(x=mt["month"], y=mt["score"], mode="lines+markers+text",
                                   text=mt["score"].round(0), textposition="top center",
                                   line=dict(color=ACCENT, width=3)))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1.container(border=True):
        st.markdown(f"**Top issues** · change vs previous {days} days")
        ti = an.top_issues(cur, prev)
        ti = ti[~is_noise(ti["issue"])]
        ti["trend"] = ti["change_pct"].apply(lambda v: "new" if pd.isna(v) else ("▲ " if v > 0 else "▼ " if v < 0 else "= ") + f"{abs(v):.0f}%")
        st.dataframe(ti[["issue", "mentions", "previous", "trend"]], hide_index=True, width="stretch",
                     column_config={"mentions": st.column_config.ProgressColumn("Mentions", format="%d", min_value=0,
                                                                                max_value=int(ti["mentions"].max() or 1)),
                                    "previous": "Previous", "trend": "Change", "issue": "Issue"})
    with c2.container(border=True):
        st.markdown("**HDBSCAN cluster distribution**")
        dist = classified(cur)["issue"].value_counts().reset_index()
        dist.columns = ["issue", "reviews"]
        noise_n = int(is_noise(cur["issue"]).sum())
        if len(dist):
            fig = px.treemap(dist, path=["issue"], values="reviews", color="reviews",
                             color_continuous_scale=["#E1F5EE", ACCENT])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        st.caption(f"{len(dist)} categories · {noise_n} reviews flagged as noise in this window")

    with st.container(border=True):
        st.markdown("**Decision panel · Top 5 issues by priority**  \n"
                    "<span class='small'>Priority = 0.4 × frequency + 0.3 × negative share + 0.3 × low rating</span>",
                    unsafe_allow_html=True)
        pt = an.priority_table(complaint_rows(cur)).head(5)
        if len(pt):
            pt["recommended_action"] = [recommend(i, "affected branches")["action"] for i in pt["issue"]]
            pt["owner"] = [recommend(i)["owner"] for i in pt["issue"]]
            pt["expected_improvement"] = [recommend(i)["expected"] for i in pt["issue"]]
            pt["campaign"] = [next((f"#{c['id']} {c['name']}" for c in CAMPAIGNS if c["id"] == match_campaign(i)), "—")
                              for i in pt["issue"]]
            st.dataframe(pt[["rank", "issue", "frequency", "avg_rating", "priority_score", "priority",
                             "recommended_action", "owner", "expected_improvement", "campaign"]],
                         hide_index=True, width="stretch",
                         column_config={"avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
                                        "priority_score": st.column_config.NumberColumn("Score", format="%.0f"),
                                        "campaign": "Linked campaign"})

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Versus competitors** · reputation score")
        comp = an.competitor_scores(an.filter_reviews(df_all, sources=sources), [ss.restaurant] + compare, cur_start, cur_end)
        comp = comp.sort_values("score")
        fig = go.Figure(go.Bar(x=comp["score"], y=comp["restaurant"], orientation="h", text=comp["score"],
                               marker_color=[ACCENT if r == ss.restaurant else GREY for r in comp["restaurant"]]))
        fig.update_layout(height=60 + 45 * len(comp), margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, width="stretch")
    with c2.container(border=True):
        st.markdown("**Customer emotions**")
        em = cur["emotion"].value_counts().reset_index()
        fig = px.bar(em, x="count", y="emotion", orientation="h", color="emotion",
                     color_discrete_map={"anger": RED, "frustration": AMBER, "disappointment": "#D4537E",
                                         "neutral": GREY, "delight": GREEN})
        fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Branches** · lowest reputation first")
        st.dataframe(an.branch_table(cur), hide_index=True, width="stretch",
                     column_config={"score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                                    "negative_pct": st.column_config.NumberColumn("Negative %", format="%.1f")})

    with st.expander("Read the reviews in this window"):
        q = st.text_input("Search text", placeholder="e.g. sambar, parking, rude")
        show = cur if not q else cur[cur["text"].str.contains(q, case=False, na=False)]
        st.dataframe(show[["review_date", "source", "branch", "rating", "issue", "emotion", "urgency", "text"]]
                     .sort_values("review_date", ascending=False), hide_index=True, width="stretch")
        st.download_button("Download CSV", show.to_csv(index=False).encode(), f"{ss.restaurant}_reviews.csv", "text/csv")

# ================================================================== 3. EARLY WARNING (MODULE 6)
elif page == PAGES[2]:
    st.markdown("**Module 6 · Reputation early-warning & resolution**  \n"
                "<span class='small'>Spikes are flagged when an issue at a branch runs well above its 4-week average. "
                "Reviews needing attention are listed most urgent first, each with a recommended fix and a drafted "
                "reply for a manager to approve.</span>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Active alerts** · last 7 days vs 4-week average")
        if len(spikes):
            sp = spikes.copy()
            sp["severity"] = ["🔴 Critical" if match_campaign(i) == 5 or r >= 3 else "🟠 High" if r >= 2 else "🟡 Watch"
                              for i, r in zip(sp["issue"], sp["ratio"])]
            cols = [c for c in ["severity", "issue", "branch", "this_week", "ratio"] if c in sp.columns]
            st.dataframe(sp[cols], hide_index=True, width="stretch",
                         column_config={"this_week": "Last 7 days",
                                        "ratio": st.column_config.NumberColumn("× 4-week avg", format="%.1f")})
            st.caption("Hygiene alerts are always critical: they pause Campaign 5 until an internal review is done.")
        else:
            st.success("No spikes this week.", icon="✅")

    pool = upto[upto["urgency"] >= 1]
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 2, 1])
    status_f = f1.selectbox("Status", ["Open", "All", "Replied", "Resolved"])
    min_u = f2.slider("Minimum urgency", 1, 10, HIGH_THRESHOLD)
    issue_f = f3.multiselect("Issue", sorted(pool["issue"].dropna().unique()), placeholder="All issues")
    per_page = f4.selectbox("Show", [10, 20, 50], index=0)

    q = pool[pool["urgency"] >= min_u]
    if status_f == "Open":
        q = q[~q["status"].isin(an.CLOSED)]
    elif status_f != "All":
        q = q[q["status"] == status_f]
    if issue_f:
        q = q[q["issue"].isin(issue_f)]
    q = q.sort_values(["urgency", "review_date"], ascending=[False, False])

    a, b, c, d = st.columns(4)
    open_all = pool[(pool["urgency"] >= HIGH_THRESHOLD) & ~pool["status"].isin(an.CLOSED)]
    a.metric("Urgent (8-10), open", int((open_all["urgency"] >= URGENT_THRESHOLD).sum()), border=True)
    b.metric("High (6-7), open", int((open_all["urgency"] < URGENT_THRESHOLD).sum()), border=True)
    c.metric("Overdue > 48 h", len(esc_overdue), border=True)
    d.metric("Resolved (all time)", int((pool["status"] == "Resolved").sum()), border=True)

    st.caption(f"{len(q)} reviews match · showing {min(per_page, len(q))}")
    for r in q.head(per_page).itertuples():
        rid = int(r.id)
        rec = recommend(r.issue, r.branch)
        if r.urgency >= URGENT_THRESHOLD:
            u = pill(f"Urgent · {r.urgency}/10", "p-urgent")
        elif r.urgency >= HIGH_THRESHOLD:
            u = pill(f"High · {r.urgency}/10", "p-high")
        else:
            u = pill(f"Medium · {r.urgency}/10")
        if r.status == "Resolved":
            u = pill("Resolved", "p-ok") + u
        age = (pd.Timestamp(as_of) - r.review_date).days
        with st.container(border=True):
            top_l, top_r = st.columns([3, 2])
            top_l.markdown(u + pill(r.issue) + pill(str(r.emotion).title()) + (pill("⚠ red flag", "p-urgent") if r.red_flag else ""),
                           unsafe_allow_html=True)
            top_r.markdown(f"<div class='small' style='text-align:right'>{r.source} · {r.branch} · "
                           f"{'★' * int(r.rating) if pd.notna(r.rating) else 'no rating'} · {r.review_date:%d %b} ({age} days ago)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='quote'>“{r.text}”</div>", unsafe_allow_html=True)
            st.info(f"**Recommended action:** {rec['action']} · Owner: {rec['owner']} · {deadline_text(rec['sla_hours'])}",
                    icon="💡")

            key = f"reply_{rid}"
            if key not in ss:
                ss[key] = r.reply_text if isinstance(r.reply_text, str) and r.reply_text else template_reply(r.restaurant, r.branch, r.issue, r.emotion, r.text)

            def ai_draft(k=key, row=r):
                ss[k], _ = draft_reply(row.restaurant, row.branch, row.issue, row.emotion, row.text, row.rating)

            st.text_area("Drafted reply (edit before approving)", key=key, height=110)
            b1, b2, b3, b4, b5 = st.columns([1.4, 1.2, 1.6, 1.1, 1.6])
            if b1.button("✅ Approve reply", key=f"ap_{rid}", type="primary"):
                db.upsert_action(rid, status="Replied" if r.status != "Resolved" else "Resolved",
                                 reply_text=ss[key], replied_at=datetime.now().isoformat(timespec="seconds"),
                                 action_text=rec["action"])
                st.toast("Reply approved — copy it and post it on " + r.source)
                st.rerun()
            b2.button("✨ AI draft", key=f"ai_{rid}", on_click=ai_draft, disabled=not groq_available(),
                      help=None if groq_available() else "Set GROQ_API_KEY to enable")
            who = b3.selectbox("Assign", TEAM, index=TEAM.index(r.assignee) if r.assignee in TEAM else 0,
                               key=f"as_{rid}", label_visibility="collapsed")
            if who != (r.assignee or "Unassigned"):
                db.upsert_action(rid, assignee=who, status="In progress" if r.status == "New" else r.status,
                                 action_text=rec["action"])
                st.rerun()
            if r.status != "Resolved" and b4.button("Resolve", key=f"rs_{rid}"):
                db.upsert_action(rid, status="Resolved", resolved_at=datetime.now().isoformat(timespec="seconds"),
                                 action_text=rec["action"])
                st.rerun()
            b5.markdown(f"<div class='small' style='padding-top:8px;text-align:right'>Status: <b>{r.status}</b></div>",
                        unsafe_allow_html=True)
    st.caption("Approving stores the reply and marks the review as replied. Posting to Google/Zomato/TripAdvisor "
               "is done by copying the reply to the platform (their reply APIs need business-account access).")

# ================================================================== 4. LOCATION & AMBIENCE (MODULE 5)
elif page == PAGES[3]:
    st.markdown("**Module 5 · Location & ambience intelligence**  \n"
                "<span class='small'>Finds reviews that mention parking, seating, ambience, cleanliness or "
                "location, and shows which branches need physical or layout fixes. A mention counts as negative "
                "when the review is rated 2★ or below (or classed negative).</span>", unsafe_allow_html=True)
    if cur.empty:
        st.info("No reviews in this window. Widen the window or change filters in the sidebar.")
        st.stop()

    flags = aspect_flags(cur)
    neg = is_negative(cur)
    rows = []
    for a in ASPECTS:
        m = flags[a]
        rows.append(dict(aspect=a, mentions=int(m.sum()), negative=int((m & neg).sum()),
                         avg_rating=cur.loc[m, "rating"].mean() if m.any() else None))
    asp = pd.DataFrame(rows)
    asp["negative_pct"] = (asp["negative"] / asp["mentions"].where(asp["mentions"] > 0) * 100).round(1)

    cols = st.columns(len(ASPECTS))
    for col, r in zip(cols, asp.itertuples()):
        col.metric(r.aspect, r.mentions,
                   f"{r.negative_pct:.0f}% negative" if pd.notna(r.negative_pct) else "no mentions",
                   delta_color="off", border=True)

    with st.container(border=True):
        st.markdown("**Negative-mention rate by branch** · % of a branch's reviews with a negative mention")
        per = pd.concat([cur[["branch"]], flags.mul(neg, axis=0)], axis=1).groupby("branch").sum()
        size = cur.groupby("branch").size()
        min_n = st.slider("Minimum reviews per branch", 1, 50, 5)
        per = per[size.reindex(per.index) >= min_n]
        if per.empty:
            st.info("No branch has enough reviews in this window. Lower the minimum or widen the window.")
        else:
            rate = (per.div(size.reindex(per.index), axis=0) * 100).round(1)
            rate = rate.loc[rate.sum(axis=1).sort_values(ascending=False).index].head(20)
            fig = px.imshow(rate, text_auto=".0f", aspect="auto", color_continuous_scale=["#FFFFFF", AMBER, RED],
                            labels=dict(color="% negative"))
            fig.update_layout(height=120 + 28 * len(rate), margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, width="stretch")

            worst = rate.idxmax(axis=1)
            fix = pd.DataFrame({"branch": rate.index, "reviews": size.reindex(rate.index).values,
                                "main_problem": worst.values,
                                "negative_pct": [rate.loc[b, a] for b, a in worst.items()]})
            fix = fix[fix["negative_pct"] > 0].head(10)
            if len(fix):
                st.markdown("**Branches to fix first**")
                st.dataframe(fix, hide_index=True, width="stretch",
                             column_config={"negative_pct": st.column_config.NumberColumn("Negative %", format="%.1f"),
                                            "main_problem": "Main problem"})

    with st.container(border=True):
        st.markdown("**Versus competitors** · negative-mention rate per aspect")
        comp_df = an.filter_reviews(df_all, sources=sources, start=cur_start, end=cur_end)
        comp_df = comp_df[comp_df["restaurant"].isin([ss.restaurant] + compare)]
        if len(comp_df):
            cf = aspect_flags(comp_df).mul(is_negative(comp_df), axis=0)
            cr = (pd.concat([comp_df[["restaurant"]], cf], axis=1).groupby("restaurant").mean() * 100).round(1)
            long = cr.reset_index().melt(id_vars="restaurant", var_name="aspect", value_name="negative_pct")
            fig = px.bar(long, x="aspect", y="negative_pct", color="restaurant", barmode="group",
                         labels={"negative_pct": "% of reviews", "aspect": ""})
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title=None,
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")

    with st.expander("Read the reviews behind an aspect"):
        pick = st.selectbox("Aspect", list(ASPECTS))
        only_neg = st.checkbox("Negative only", value=True)
        m = flags[pick] & (neg if only_neg else True)
        st.dataframe(cur.loc[m, ["review_date", "source", "branch", "rating", "issue", "text"]]
                     .sort_values("review_date", ascending=False), hide_index=True, width="stretch")

# ================================================================== 5. CAMPAIGNS (360-DEGREE)
elif page == PAGES[4]:
    st.markdown("**Digital marketing · 360-degree campaigns tied to issue clusters**  \n"
                "<span class='small'>Each campaign answers one HDBSCAN cluster. Success is measured the same way the "
                "issue was found: the cluster's share of classified reviews should fall on the next pipeline "
                "re-run.</span>", unsafe_allow_html=True)

    cls_cur, cls_prev = classified(cur), classified(prev)
    if cls_cur.empty:
        st.info("No classified reviews in this window. Widen the window or change filters in the sidebar.")
        st.stop()
    cid_cur = cls_cur["issue"].map(match_campaign)
    cid_prev = cls_prev["issue"].map(match_campaign)
    cid_upto = upto["issue"].map(match_campaign)
    spike_cids = set(spikes["issue"].map(match_campaign).dropna()) if len(spikes) else set()

    rows = []
    for c in CAMPAIGNS:
        share_now = (cid_cur == c["id"]).mean() * 100
        share_prev = (cid_prev == c["id"]).mean() * 100 if len(cls_prev) else None
        change = None if not share_prev else (share_now - share_prev) / share_prev * 100
        wk = weekly_counts(upto[cid_upto == c["id"]], as_of)
        base = sum(wk[:-1]) / 4
        if c["mode"] == "risk":
            status = "⏸ Pause — internal review" if c["id"] in spike_cids else "▶ Run (monitor weekly)"
        elif c["mode"] == "gated":
            dropping = wk[-1] < base and c["id"] not in spike_cids
            status = "▶ Ready to launch" if dropping else "⏸ Hold until complaints drop"
        else:
            status = "▶ Launch now"
        rows.append(dict(rank=c["id"], campaign=c["name"], cluster=c["theme"],
                         reviews=int((cid_cur == c["id"]).sum()), share=round(share_now, 1),
                         change=None if change is None else round(change, 0),
                         last_week=wk[-1], avg_4w=round(base, 1), status=status))
    tbl = pd.DataFrame(rows).sort_values("reviews", ascending=False)

    top = tbl.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Reviews covered by the 5 campaigns", f"{tbl['share'].sum():.0f}%",
              f"of {len(cls_cur):,} classified reviews", delta_color="off", border=True)
    k2.metric("Largest cluster", top["cluster"], f"{top['share']:.1f}% share", delta_color="off", border=True)
    k3.metric("Campaigns on hold", int(tbl["status"].str.startswith("⏸").sum()), border=True)

    with st.container(border=True):
        st.markdown(f"**Campaign status** · share of classified reviews, last {days} days vs previous {days}")
        st.dataframe(tbl.drop(columns="rank"), hide_index=True, width="stretch",
                     column_config={"share": st.column_config.ProgressColumn("Share %", min_value=0,
                                                                             max_value=max(1.0, float(tbl["share"].max())),
                                                                             format="%.1f%%"),
                                    "change": st.column_config.NumberColumn("Change %", format="%+.0f%%"),
                                    "last_week": "Last 7 days", "avg_4w": "4-week avg / wk"})
        st.caption("Campaign 4 (wait time) launches only after the operational fix shows a measurable drop — "
                   "a speed promise followed by a slow visit backfires. Campaign 5 (hygiene) pauses on any spike.")

    with st.container(border=True):
        st.markdown("**Weekly complaint volume per campaign cluster** · last 5 weeks")
        trend = pd.DataFrame({c["name"]: weekly_counts(upto[cid_upto == c["id"]], as_of) for c in CAMPAIGNS},
                             index=[f"W-{i}" if i else "This week" for i in range(4, -1, -1)])
        fig = px.line(trend.reset_index().melt(id_vars="index", var_name="campaign", value_name="reviews"),
                      x="index", y="reviews", color="campaign", markers=True)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, legend_title=None,
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1.container(border=True):
        st.markdown("**Campaign briefs**")
        for c in sorted(CAMPAIGNS, key=lambda c: -int(tbl.loc[tbl["rank"] == c["id"], "reviews"].iloc[0])):
            with st.expander(f"#{c['id']} {c['name']} — {c['theme']}"):
                st.markdown(f"<div class='quote'>“{c['headline']}”</div>", unsafe_allow_html=True)
                st.markdown(" ".join(pill(ch) for ch in c["channels"]), unsafe_allow_html=True)
                st.markdown(f"<span class='small'>KPI: {c['kpi']}</span>", unsafe_allow_html=True)
                ex = cls_cur[cid_cur == c["id"]].sort_values("rating").head(3)
                for t in ex["text"]:
                    st.markdown(f"<div class='small'>• {str(t)[:220]}</div>", unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown("**360° channel matrix**")
        mat = pd.DataFrame({f"#{c['id']}": ["✅" if ch in c["channels"] else "" for ch in CHANNELS]
                            for c in sorted(CAMPAIGNS, key=lambda c: c["id"])}, index=CHANNELS)
        st.dataframe(mat, width="stretch")

    unmatched = cls_cur.loc[cid_cur.isna(), "issue"].value_counts()
    if len(unmatched):
        with st.expander(f"Clusters without a campaign ({len(unmatched)})"):
            st.dataframe(unmatched.rename("reviews"), width="stretch")
            st.caption("These are usually positive or low-volume clusters. Add keywords to CAMPAIGNS in main.py "
                       "if a new complaint cluster appears after a re-run.")

# ================================================================== 6. MONTHLY REPORT
elif page == PAGES[5]:
    rest_all = df_all[df_all["restaurant"] == ss.restaurant]
    months = sorted(rest_all["review_date"].dt.to_period("M").astype(str).unique(), reverse=True)
    c1, c2 = st.columns([2, 1])
    month = c1.selectbox("Month", months, format_func=lambda m: pd.Period(m).strftime("%B %Y"))
    rkey = f"report_{ss.restaurant}_{month}"
    if c2.button("🔄 Generate / refresh report", type="primary", width="stretch") or rkey not in ss:
        with st.spinner("Building report..."):
            rep = build_report(an.filter_reviews(df_all, sources=sources), ss.restaurant, month)
            pdf, path = save_report(rep)
            ss[rkey] = (rep, pdf)
    rep, pdf = ss[rkey]
    k, rp = rep["kpis"], rep["reply"]

    with st.container(border=True):
        t1, t2 = st.columns([4, 1])
        t1.markdown(f"### {rep['restaurant']} · {rep['label']} report")
        t1.markdown(f"<p class='small'>Generated {date.today():%d %b %Y} · {k['reviews']} reviews from "
                    f"{len(rep['sources'])} sources ({', '.join(f'{s} {n}' for s, n in rep['sources'].items())})"
                    f" · summary by {rep['engine']}</p>", unsafe_allow_html=True)
        t2.download_button("⬇️ PDF", pdf, f"{rep['restaurant'].replace(' ', '_')}_{month}.pdf", "application/pdf",
                           width="stretch")
        st.markdown("**Summary**")
        st.write(rep["summary"])
        st.markdown("**Top 3 actions for next month**")
        for i, a in enumerate(rep["actions"], 1):
            cls = {1: "p-urgent", 2: "p-high", 3: "p-ok"}[i]
            st.markdown(f"{pill(str(i), cls)} **{a['action']}** <span class='small'>· {a['issue']} · "
                        f"Owner: {a['owner']} · Target: {a['expected']}</span>", unsafe_allow_html=True)
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Reputation score", k["score"], None if k["score_delta"] is None else f"{k['score_delta']:+.1f} pts", border=True)
        x2.metric("Reply rate (negative reviews)", "-" if rp["reply_rate"] is None else f"{rp['reply_rate']:.0f}%", border=True)
        x3.metric("Median reply time", "-" if rp["avg_reply_hours"] is None else f"{rp['avg_reply_hours']:.0f} h", border=True)
        x4.metric("Escalations resolved", f"{rp['resolved']}/{rp['escalations']}", border=True)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Complaint themes vs last month**")
        iss = rep["issues"]
        iss = iss[~is_noise(iss["issue"])].head(8)
        fig = go.Figure()
        fig.add_bar(y=iss["issue"], x=iss["previous"], name="Last month", orientation="h", marker_color=GREY)
        fig.add_bar(y=iss["issue"], x=iss["mentions"], name="This month", orientation="h", marker_color=ACCENT)
        fig.update_layout(height=340, barmode="group", margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with c2.container(border=True):
        st.markdown("**Branches**")
        st.dataframe(rep["branches"], hide_index=True, width="stretch", height=340)

    with st.expander("Previously generated reports"):
        with db.connect() as con:
            past = pd.read_sql_query("SELECT restaurant, month, created_at, pdf_path FROM reports ORDER BY month DESC", con)
        st.dataframe(past, hide_index=True, width="stretch")
