"""Empathy Engine — Streamlit front end.   Run:  streamlit run app.py

Screens: 1. Search   2. Overview   3. Action center   4. Monthly report
"""
import difflib
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import analytics as an
from engine import db
from engine.config import (HIGH_THRESHOLD, POSITIVE_ISSUE, RAW_DIR, SOURCES, TEAM, URGENT_THRESHOLD,
                           canonical_chain_name)
from engine.importer import import_file
from engine.playbook import deadline_text, recommend
from engine.replies import draft_reply, groq_available, template_reply
from engine.report import build_report, save_report

st.set_page_config(page_title="Empathy Engine", page_icon="💬", layout="wide")
db.init_db()

GREEN, RED, GREY, AMBER, ACCENT = "#1D9E75", "#D85A30", "#B4B2A9", "#BA7517", "#0F6E56"
SENT_COLORS = {"negative": RED, "neutral": GREY, "positive": GREEN}
PAGES = ["🔍 1. Search", "📊 2. Overview", "🚨 3. Action center", "📄 4. Monthly report"]

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

To try the app with sample data first: `python tools/make_sample_data.py` then the import command above.
""")
    source = st.selectbox("Source", SOURCES)
    ups = st.file_uploader("Review files (CSV or Excel)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
    ref = st.date_input("Date the files were scraped (for '2 months ago' dates)", date.today())
    if ups and st.button("Import", type="primary"):
        for u in ups:
            path = RAW_DIR / source.lower() / u.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(u.getbuffer())
            st.write(import_file(path, source, ref))
        st.rerun()
    st.stop()

df_all["restaurant"] = df_all["restaurant"].map(canonical_chain_name)
restaurants = sorted(set(df_all["restaurant"]) | {"Namma Veedu Vasantha Bhavan"})
data_end = df_all["review_date"].max().date()
data_start = df_all["review_date"].min().date()
ss = st.session_state
ss.setdefault("restaurant", "Geetham" if "Geetham" in restaurants else restaurants[0])
ss["restaurant"] = canonical_chain_name(ss["restaurant"])
if ss["restaurant"] not in restaurants:
    ss["restaurant"] = restaurants[0]
ss.setdefault("page", PAGES[0])

with st.sidebar:
    st.markdown("### 💬 Empathy Engine")
    st.selectbox("Restaurant", restaurants, key="restaurant")
    rest_df = df_all[df_all["restaurant"] == ss.restaurant]
    branches = st.multiselect("Branches", sorted(rest_df["branch"].unique()), placeholder="All branches")
    sources = st.multiselect("Sources", SOURCES, default=SOURCES)
    days = st.select_slider("Window", options=[7, 14, 30, 60, 90, 180], value=30, format_func=lambda d: f"Last {d} days")
    as_of = st.date_input("As of", data_end, min_value=data_start, max_value=data_end)
    compare = st.multiselect("Compare with", [r for r in restaurants if r != ss.restaurant],
                             default=[r for r in ("A2B", "Sangeetha") if r in restaurants and r != ss.restaurant])
    st.divider()
    st.caption(f"Data: {len(df_all):,} reviews · {data_start:%d %b %Y} – {data_end:%d %b %Y}")
    st.caption("Replies: " + ("Groq LLM ✅" if groq_available() else "templates (set GROQ_API_KEY for AI replies)"))

(cur_start, cur_end), (prev_start, prev_end) = an.windows(as_of, days)
scope = an.filter_reviews(df_all, ss.restaurant, branches or None, sources)
cur = an.filter_reviews(scope, start=cur_start, end=cur_end)
prev = an.filter_reviews(scope, start=prev_start, end=prev_end)
spikes = an.spikes(an.filter_reviews(scope, end=as_of), as_of)
esc_open, esc_overdue = an.open_escalations(an.filter_reviews(scope, end=as_of), as_of)

# ------------------------------------------------------------------ header + navigation
h1, h2 = st.columns([3, 2])
h1.markdown(f"## 💬 Empathy Engine")
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
        counts = cur["source"].value_counts()
        pills = "".join(pill(f"✓ {s} · {counts.get(s, 0)}", "p-ok") if counts.get(s, 0) else pill(f"{s} · no data", "p-muted")
                        for s in SOURCES)
        pills += pill("＋ YouTube · planned", "p-muted") + pill("🔒 Instagram · connect account", "p-muted")
        st.markdown(f"<div style='text-align:center'><p class='small'>Sources</p>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='small' style='text-align:center'>📅 Last {days} days · {cur_start:%d %b} – "
                    f"{cur_end:%d %b %Y} &nbsp;&nbsp; ⚖️ Compare with: {', '.join(compare) or '—'}</p>",
                    unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Reviews collected** — selected window")
        mx = max(1, counts.max() if len(counts) else 1)
        for s in SOURCES:
            a, b, c = st.columns([1.2, 6, 0.8])
            a.write(s)
            b.progress(int(counts.get(s, 0) / mx * 100))
            c.write(f"**{counts.get(s, 0)}**")
        st.button("See the overview →", on_click=go_overview)

    with st.expander("📥 Add or update review files (Google / Zomato / TripAdvisor)"):
        c1, c2 = st.columns(2)
        src = c1.selectbox("Source of these files", SOURCES)
        ref = c2.date_input("Date the files were scraped", date.today(),
                            help="Used to convert Google's '3 weeks ago' style dates into real dates.")
        ups = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        if ups and st.button("Import files", type="primary"):
            for u in ups:
                path = RAW_DIR / src.lower() / u.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(u.getbuffer())
                r = import_file(path, src, ref)
                (st.error if r["message"].startswith("ERROR") else st.success)(
                    f"{u.name}: {r['added']} new of {r['read']} rows. {r['message']}")
            st.button("Refresh")
        st.caption("Duplicates are skipped automatically, so re-uploading a file is safe. "
                   "If your file has an issue/cluster label column (from HDBSCAN), it is kept.")
        runs = db.load_runs()
        if len(runs):
            st.dataframe(runs[["started_at", "source", "file_name", "rows_read", "rows_added", "message"]],
                         hide_index=True, width="stretch")

    with st.expander("🗂️ Data coverage — all restaurants"):
        cov = df_all.pivot_table(index="restaurant", columns="source", values="id", aggfunc="count", fill_value=0)
        cov["Total"] = cov.sum(axis=1)
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
        mt = an.monthly_trend(an.filter_reviews(scope, end=as_of))
        fig = go.Figure(go.Scatter(x=mt["month"], y=mt["score"], mode="lines+markers+text",
                                   text=mt["score"].round(0), textposition="top center",
                                   line=dict(color=ACCENT, width=3)))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown(f"**Top issues** · change vs previous {days} days")
        ti = an.top_issues(cur, prev)
        ti["trend"] = ti["change_pct"].apply(lambda v: "new" if pd.isna(v) else ("▲ " if v > 0 else "▼ " if v < 0 else "= ") + f"{abs(v):.0f}%")
        st.dataframe(ti[["issue", "mentions", "previous", "trend"]], hide_index=True, width="stretch",
                     column_config={"mentions": st.column_config.ProgressColumn("Mentions", format="%d", min_value=0,
                                                                                max_value=int(ti["mentions"].max() or 1)),
                                    "previous": "Previous", "trend": "Change", "issue": "Issue"})

    with st.container(border=True):
        st.markdown("**Decision panel · Top 5 issues by priority**  \n"
                    "<span class='small'>Priority = 0.4 × frequency + 0.3 × negative share + 0.3 × low rating</span>",
                    unsafe_allow_html=True)
        pt = an.priority_table(cur).head(5)
        if len(pt):
            pt["recommended_action"] = [recommend(i, "affected branches")["action"] for i in pt["issue"]]
            pt["owner"] = [recommend(i)["owner"] for i in pt["issue"]]
            pt["expected_improvement"] = [recommend(i)["expected"] for i in pt["issue"]]
            st.dataframe(pt[["rank", "issue", "frequency", "avg_rating", "priority_score", "priority",
                             "recommended_action", "owner", "expected_improvement"]],
                         hide_index=True, width="stretch",
                         column_config={"avg_rating": st.column_config.NumberColumn("Avg rating", format="%.2f"),
                                        "priority_score": st.column_config.NumberColumn("Score", format="%.0f")})

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
        st.markdown("**Emotion analysis**")
        em = cur.loc[cur["emotion"] != "neutral", "emotion"].value_counts().reset_index()
        fig = px.bar(em, x="count", y="emotion", orientation="h", color="emotion",
                     color_discrete_map={"anger": RED, "frustration": AMBER, "disappointment": "#D4537E",
                                         "delight": GREEN},
                     labels={"emotion": "Emotion", "count": "Reviews"})
        fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
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

# ================================================================== 3. ACTION CENTER
elif page == PAGES[2]:
    st.markdown("<p class='small'>Reviews needing attention, most urgent first. Each comes with a recommended fix "
                "and a drafted reply for a manager to approve.</p>", unsafe_allow_html=True)
    pool = an.filter_reviews(scope, end=as_of)
    pool = pool[pool["urgency"] >= 1]
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 2, 1])
    status_f = f1.selectbox("Status", ["Open", "All", "Replied", "Resolved"])
    min_u = f2.slider("Minimum urgency", 1, 10, HIGH_THRESHOLD)
    issue_f = f3.multiselect("Issue", sorted(pool["issue"].unique()), placeholder="All issues")
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
            top_l.markdown(u + pill(r.issue) + pill(r.emotion.title()) + (pill("⚠ red flag", "p-urgent") if r.red_flag else ""),
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

# ================================================================== 4. MONTHLY REPORT
elif page == PAGES[3]:
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
        iss = rep["issues"].head(8)
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
