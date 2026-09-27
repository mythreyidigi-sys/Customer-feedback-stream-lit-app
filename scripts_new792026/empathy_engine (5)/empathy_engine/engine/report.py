"""Monthly report: summary, top 3 actions, KPIs and a downloadable PDF."""
from io import BytesIO
import re

import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import analytics as an
from . import db
from .config import CHAIN_ALIASES, POSITIVE_ISSUE, REPORT_DIR, SERVQUAL_RESPONSES_PATH
from .playbook import recommend
from .replies import monthly_summary

ACCENT = colors.HexColor("#0F6E56")
GREY = colors.HexColor("#B4B2A9")
SERVQUAL_ISSUES = {
    "Tangibles": ["Ambience & Location", "Poor Experience / Hygiene Complaints"],
    "Reliability": ["Food Quantity & Value for Money", "Food Quality & Branch Consistency", "Service Quality"],
    "Responsiveness": ["Slow Service & Staff Negligence", "Food Variety & Fast Service"],
    "Assurance": ["Poor Experience / Hygiene Complaints", "Pricing & Staff Performance"],
    "Empathy": ["Pricing & Staff Performance", "Service Quality"],
}
SERVQUAL_QUESTIONS = {
    "Tangibles": ["cleanlinessofrestaurant", "overallambience"],
    "Reliability": ["qualityoffood", "consistencyofqualityandtasteoffood", "properbillingforordersserved"],
    "Responsiveness": ["arrivaloffoodontime"],
    "Assurance": ["staffbehaviourandappearance"],
    "Empathy": ["personalisedattentivenesstospecialneeds"],
}


def build_report(df, restaurant, month: str) -> dict:
    """month = 'YYYY-MM'."""
    p = pd.Period(month, "M")
    start, end = p.start_time, p.end_time.normalize()
    prev = p - 1
    rest = an.filter_reviews(df, restaurant)
    cur = an.filter_reviews(rest, start=start, end=end)
    prv = an.filter_reviews(rest, start=prev.start_time, end=prev.end_time.normalize())

    k = an.kpis(cur, prv, end)
    issues = an.top_issues(cur, prv)
    prio = an.priority_table(cur)
    sp = an.spikes(rest, min(end, rest["review_date"].max()) if len(rest) else end)
    rs = an.reply_stats(cur)
    by_src = cur["source"].value_counts().to_dict()

    rising = issues.dropna(subset=["change_pct"]).sort_values("change_pct")
    facts = {
        "reviews": k["reviews"], "sources": by_src, "score": k["score"], "score_delta": k["score_delta"],
        "negative_share_pct": round(k["neg_share"], 1),
        "top_issue": issues["issue"].iat[0] if len(issues) else None,
        "top_issue_count": int(issues["mentions"].iat[0]) if len(issues) else 0,
        "biggest_rise": (f"{rising['issue'].iat[-1]} ({rising['change_pct'].iat[-1]:+.0f}%)"
                         if len(rising) and rising["change_pct"].iat[-1] > 0 else None),
        "biggest_fall": (rising["issue"].iat[0] if len(rising) and rising["change_pct"].iat[0] < 0 else None),
        "spike": (f"{sp['issue'].iat[0]} at {sp['branch'].iat[0]} ({sp['ratio'].iat[0]:.1f}x its weekly average)"
                  if len(sp) else None),
    }
    summary, engine = monthly_summary(restaurant, p.strftime("%B %Y"), facts)

    actions = []
    for _, row in prio.head(3).iterrows():
        d = cur[(cur["issue"] == row["issue"]) & (cur["sentiment"] != "positive")]
        branch = d["branch"].mode().iat[0] if len(d) else "all branches"
        rec = recommend(row["issue"], branch)
        actions.append({"issue": row["issue"], "action": rec["action"], "owner": rec["owner"],
                        "expected": rec["expected"], "priority": str(row["priority"])})

    return {"restaurant": restaurant, "month": month, "label": p.strftime("%B %Y"), "kpis": k, "facts": facts,
            "summary": summary, "engine": engine, "actions": actions, "issues": issues, "priority": prio,
            "branches": an.branch_table(cur), "reply": rs, "sources": by_src}


def _fmt(v, suffix="", nd=0):
    return "-" if v is None or pd.isna(v) else f"{v:.{nd}f}{suffix}"


def build_pdf(r: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{r['restaurant']} {r['label']} review report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], alignment=0, fontSize=18, textColor=ACCENT, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    k, rp = r["kpis"], r["reply"]
    src = " · ".join(f"{s} {n}" for s, n in r["sources"].items())
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E1F5EE")),
                       ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP")])

    el = [Paragraph(f"{r['restaurant']} · {r['label']} report", h1),
          Paragraph(f"{k['reviews']} reviews ({src}) · generated by Empathy Engine", small),
          Paragraph("Summary", h2), Paragraph(r["summary"], body), Paragraph("Key numbers", h2)]
    kt = [["Reputation score", "Change", "Avg rating", "Negative share", "Reply rate", "Median reply time", "Escalations resolved"],
          [_fmt(k["score"], "/100", 1), _fmt(k["score_delta"], " pts", 1), _fmt(k["avg_rating"], "", 2),
           _fmt(k["neg_share"], "%", 1), _fmt(rp["reply_rate"], "%"), _fmt(rp["avg_reply_hours"], " h"),
           f"{rp['resolved']}/{rp['escalations']}"]]
    el.append(Table(kt, style=grid))

    el.append(Paragraph("Top 3 actions for next month", h2))
    at = [["#", "Issue", "Action", "Owner", "Expected result"]] + [
        [i + 1, Paragraph(a["issue"], body), Paragraph(a["action"], body), a["owner"], Paragraph(a["expected"], body)]
        for i, a in enumerate(r["actions"])]
    el.append(Table(at, colWidths=[8 * mm, 38 * mm, 62 * mm, 26 * mm, 40 * mm], style=grid))

    iss = r["issues"].head(8)
    if len(iss):
        el.append(Paragraph("Complaint themes", h2))
        d = Drawing(170 * mm, 8 * mm * len(iss) + 10)
        ch = HorizontalBarChart()
        ch.x, ch.y, ch.width, ch.height = 62 * mm, 5, 100 * mm, 8 * mm * len(iss)
        ch.data = [list(iss["mentions"][::-1])]
        ch.categoryAxis.categoryNames = [s[:38] for s in iss["issue"][::-1]]
        ch.categoryAxis.labels.fontSize = 7.5
        ch.categoryAxis.labels.fontName = ch.valueAxis.labels.fontName = "Helvetica"
        ch.valueAxis.labels.fontSize = 7
        ch.valueAxis.valueMin = 0
        ch.bars[0].fillColor = ACCENT
        ch.bars[0].strokeColor = None
        d.add(ch)
        el.append(d)
        it = [["Issue", "Mentions", "Last month", "Change"]] + [
            [row.issue, int(row.mentions), int(row.previous), _fmt(row.change_pct, "%")] for row in iss.itertuples()]
        el.append(Spacer(1, 4))
        el.append(Table(it, colWidths=[80 * mm, 25 * mm, 25 * mm, 25 * mm], style=grid))

    b = r["branches"]
    if len(b):
        el.append(Paragraph("Branches (lowest score first)", h2))
        bt = [["Branch", "Reviews", "Avg rating", "Negative %", "Score", "Top complaint"]] + [
            [row.branch, row.reviews, _fmt(row.avg_rating, "", 2), _fmt(row.negative_pct, "%", 1), _fmt(row.score, "", 1),
             Paragraph(str(row.top_complaint), body)] for row in b.head(15).itertuples()]
        el.append(Table(bt, colWidths=[35 * mm, 18 * mm, 20 * mm, 20 * mm, 15 * mm, 66 * mm], style=grid))
    el.append(Spacer(1, 8))
    el.append(Paragraph("Reputation score = 60% average rating (rescaled 1-5 to 0-100) + 40% net sentiment. "
                        "Priority = 0.4 × frequency + 0.3 × negative share + 0.3 × low rating.", small))
    doc.build(el)
    return buf.getvalue()


def save_report(r: dict) -> tuple[bytes, str]:
    pdf = build_pdf(r)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{r['restaurant'].replace(' ', '_')}_{r['month']}.pdf"
    path.write_bytes(pdf)
    db.save_report(r["restaurant"], r["month"], r["summary"], path)
    return pdf, str(path)


def _canonical_restaurant(value: str) -> str:
    normalised = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    for alias in sorted(CHAIN_ALIASES, key=len, reverse=True):
        alias_normalised = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
        if re.search(rf"\b{re.escape(alias_normalised)}\b", normalised):
            return CHAIN_ALIASES[alias]
    return normalised.title()


def servqual_restaurants() -> list[str]:
    if not SERVQUAL_RESPONSES_PATH.is_file():
        raise FileNotFoundError(f"SERVQUAL response workbook not found: {SERVQUAL_RESPONSES_PATH}")
    responses = pd.read_excel(SERVQUAL_RESPONSES_PATH)
    restaurant_col = next((c for c in responses if "restaurantname" in re.sub(r"[^a-z]", "", c.lower())), None)
    if restaurant_col is None:
        raise ValueError("The SERVQUAL workbook has no restaurant-name column.")
    return sorted({
        _canonical_restaurant(value) for value in responses[restaurant_col].dropna()
        if str(value).strip()
    })


def load_servqual_scores(restaurant: str) -> pd.DataFrame:
    if not SERVQUAL_RESPONSES_PATH.is_file():
        raise FileNotFoundError(f"SERVQUAL response workbook not found: {SERVQUAL_RESPONSES_PATH}")
    responses = pd.read_excel(SERVQUAL_RESPONSES_PATH)
    restaurant_col = next((c for c in responses if "restaurantname" in re.sub(r"[^a-z]", "", c.lower())), None)
    timestamp_col = next((c for c in responses if c.strip().lower() == "timestamp"), None)
    if restaurant_col is None:
        raise ValueError("The SERVQUAL workbook has no restaurant-name column.")

    responses = responses[responses[restaurant_col].map(_canonical_restaurant) == _canonical_restaurant(restaurant)].copy()
    if responses.empty:
        raise ValueError(f"No SERVQUAL survey responses were found for {restaurant}.")

    normalised_columns = {c: re.sub(r"[^a-z]", "", c.lower()) for c in responses.columns}
    dimension_rows = []
    respondent_mask = pd.Series(False, index=responses.index)
    valid_dates = pd.to_datetime(responses[timestamp_col], errors="coerce") if timestamp_col else pd.Series(dtype="datetime64[ns]")
    for dimension, questions in SERVQUAL_QUESTIONS.items():
        expectations, perceptions, gaps = [], [], []
        item_count = 0
        for question in questions:
            expectation_col = next((c for c, normalised in normalised_columns.items()
                                    if question in normalised and "expectation" in normalised), None)
            perception_col = next((c for c, normalised in normalised_columns.items()
                                   if question in normalised and "actual" in normalised), None)
            if expectation_col is None or perception_col is None:
                continue
            expectation = pd.to_numeric(responses[expectation_col], errors="coerce")
            perception = pd.to_numeric(responses[perception_col], errors="coerce")
            paired = pd.concat([expectation, perception], axis=1).dropna()
            paired = paired[(paired.iloc[:, 0].between(1, 7)) & (paired.iloc[:, 1].between(1, 7))]
            if paired.empty:
                continue
            respondent_mask.loc[paired.index] = True
            expectations.extend(paired.iloc[:, 0].tolist())
            perceptions.extend(paired.iloc[:, 1].tolist())
            gaps.extend((paired.iloc[:, 1] - paired.iloc[:, 0]).tolist())
            item_count += 1
        if not expectations:
            continue
        dimension_rows.append({
            "dimension": dimension,
            "mean_expectation": round(sum(expectations) / len(expectations), 2),
            "mean_perception": round(sum(perceptions) / len(perceptions), 2),
            "mean_gap": round(sum(gaps) / len(gaps), 2),
            "n_items": item_count,
        })

    if not dimension_rows:
        raise ValueError(f"No complete 1–7 expectation/actual responses were found for {restaurant}.")
    dates = valid_dates.loc[respondent_mask].dropna() if timestamp_col else pd.Series(dtype="datetime64[ns]")
    scores = pd.DataFrame(dimension_rows)
    scores["respondents"] = int(respondent_mask.sum())
    scores["survey_start"] = dates.min().strftime("%d %b %Y") if len(dates) else "Unknown"
    scores["survey_end"] = dates.max().strftime("%d %b %Y") if len(dates) else "Unknown"
    return scores.sort_values("mean_gap")


def servqual_review_frequency(reviews: pd.DataFrame) -> pd.DataFrame:
    complaints = reviews[reviews["sentiment"] != "positive"]
    counts = complaints["issue"].value_counts()
    return pd.DataFrame([
        {"dimension": dimension, "issues": ", ".join(issues),
         "review_count": int(sum(counts.get(issue, 0) for issue in issues))}
        for dimension, issues in SERVQUAL_ISSUES.items()
    ])


def build_servqual_pdf(scores: pd.DataFrame, frequency: pd.DataFrame, restaurant: str, month: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{restaurant} SERVQUAL survey report")
    ss = getSampleStyleSheet()
    title = ParagraphStyle("servqual_title", parent=ss["Title"], alignment=0, fontSize=18,
                           textColor=ACCENT, spaceAfter=4)
    heading = ParagraphStyle("servqual_heading", parent=ss["Heading2"], fontSize=12,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("servqual_body", parent=ss["BodyText"], fontSize=9, leading=12)
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E1F5EE")),
                       ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")])
    label = pd.Period(month, "M").strftime("%B %Y")
    survey_meta = scores.iloc[0]
    elements = [Paragraph(f"{restaurant} · SERVQUAL survey", title),
                Paragraph(f"Review report: {label} · {int(survey_meta['respondents'])} survey respondents · "
                          f"survey dates: {survey_meta['survey_start']} – {survey_meta['survey_end']}",
                          body), Paragraph("Expectation vs perception", heading)]

    chart_data = scores.sort_values("mean_gap")
    drawing = Drawing(174 * mm, 70 * mm)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 30, 8, 440, 165
    chart.data = [chart_data["mean_expectation"].tolist(), chart_data["mean_perception"].tolist()]
    chart.categoryAxis.categoryNames = chart_data["dimension"].tolist()
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 1, 7, 1
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor, chart.bars[1].fillColor = GREY, ACCENT
    chart.bars[0].strokeColor = chart.bars[1].strokeColor = None
    drawing.add(chart)
    elements.extend([drawing, Paragraph("SERVQUAL gap (perception − expectation)", heading)])

    gap_drawing = Drawing(174 * mm, 55 * mm)
    gap_chart = HorizontalBarChart()
    gap_chart.x, gap_chart.y, gap_chart.width, gap_chart.height = 120, 5, 330, 120
    gap_chart.data = [chart_data["mean_gap"].tolist()]
    gap_chart.categoryAxis.categoryNames = chart_data["dimension"].tolist()
    gap_chart.categoryAxis.labels.fontSize = 7
    gap_min = float(chart_data["mean_gap"].min())
    gap_max = float(chart_data["mean_gap"].max())
    gap_chart.valueAxis.valueMin = min(0, gap_min) - 0.25
    gap_chart.valueAxis.valueMax = max(0, gap_max) + 0.25
    gap_chart.valueAxis.labels.fontSize = 7
    gap_chart.bars[0].fillColor = colors.HexColor("#D85A30")
    gap_chart.bars[0].strokeColor = None
    gap_drawing.add(gap_chart)
    elements.extend([gap_drawing, Paragraph("Gap and uploaded-review themes", heading)])

    score_rows = [["Dimension", "Items", "Expectation", "Perception", "Gap"]]
    score_rows.extend([[row.dimension, row.n_items, f"{row.mean_expectation:.2f}", f"{row.mean_perception:.2f}",
                        f"{row.mean_gap:+.2f}"] for row in chart_data.itertuples()])
    elements.append(Table(score_rows, colWidths=[45 * mm, 18 * mm, 34 * mm, 34 * mm, 25 * mm], style=grid))
    freq_rows = [["Dimension", "Aligned review themes", "Reviews"]]
    freq_rows.extend([[row.dimension, Paragraph(row.issues, body), row.review_count]
                      for row in frequency.itertuples()])
    elements.extend([Spacer(1, 5), Table(freq_rows, colWidths=[35 * mm, 105 * mm, 20 * mm], style=grid),
                     Spacer(1, 6), Paragraph("Negative gaps indicate that perceived service fell below expectations. "
                     "Review-theme counts are based on the selected restaurant and report month.", body)])
    doc.build(elements)
    return buf.getvalue()
