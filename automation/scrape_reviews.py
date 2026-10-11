"""Scrape the last N months of reviews (text, star rating, date) from Google Maps, Zomato and TripAdvisor.

Usage:
    python automation/scrape_reviews.py                       # everything, last 6 months
    python automation/scrape_reviews.py --sources google --chains A2B --months 3 --headful

Output (one file per source+chain, overwritten on each run):
    Analysis/data/raw_reviews_auto/<Google|Zomato|TripAdvisor>/<chain>.csv
    columns: restaurant, branch, source, review, rating, date
A dated copy is also kept in Analysis/data/raw_reviews_auto/_archive/<YYYY-MM-DD>/.

Notes: these sites have no free review API, so this drives a real browser. Page layouts and anti-bot
rules change; selectors below are best-effort and failures are logged per target without stopping the run.
Check each site's terms of use before scraping at scale.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dateutil import parser as dateparser

ROOT = Path(__file__).resolve().parent.parent
TARGETS_FILE = Path(__file__).resolve().parent / "targets.json"
OUT_DIR = ROOT / "Analysis" / "data" / "raw_reviews_auto"
SOURCES = {"google": "Google", "zomato": "Zomato", "tripadvisor": "TripAdvisor"}
COLUMNS = ["restaurant", "branch", "source", "review", "rating", "date"]

CHAIN_KEYWORDS = {
    "A2B": ["adyar ananda", "adyar anandha", "a2b"],
    "Geetham": ["geetham"],
    "Namma Veedu Vasantha Bhavan": ["vasant"],
    "Sangeetha": ["sangeetha"],
    "Saravana Bhavan": ["saravana"],
    "Sree Annapoorna": ["annapoorna", "annapurna"],
}

log = logging.getLogger("scraper")
WINDOW_END = date.max  # newest review date to keep; set from --end


# ----------------------------------------------------------------------------- helpers
def pause(lo: float = 1.0, hi: float = 2.5) -> None:
    time.sleep(random.uniform(lo, hi))


def parse_date(text: str, today: date | None = None) -> date | None:
    """Handles '3 weeks ago', 'a month ago', 'yesterday', '2026-03-04', 'May 3, 2026', 'May 2026'."""
    today = today or date.today()
    if not text:
        return None
    s = re.sub(r"^(edited|written|reviewed|date of visit:?|visited)\s+", "", text.strip().lower()).strip()
    if s in {"today", "just now"} or re.fullmatch(r"(an?|\d+)\s+(second|minute|hour)s?\s+ago", s):
        return today
    if s == "yesterday":
        return today - timedelta(days=1)
    m = re.fullmatch(r"(an?|\d+)\s+(day|week|month|year)s?\s+ago", s)
    if m:
        n = 1 if m.group(1) in ("a", "an") else int(m.group(1))
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)] * n
        return today - timedelta(days=days)
    try:
        return dateparser.parse(s, default=datetime(today.year, today.month, 15), dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def parse_rating(text: str) -> float | None:
    m = re.search(r"(\d(?:\.\d)?)", text or "")
    if not m:
        return None
    value = float(m.group(1))
    if "bubbles" in (text or "").lower() and value > 5:  # '40' style
        value /= 10
    return value if 1 <= value <= 5 else None


def accept_consent(page) -> None:
    for label in ("Accept all", "I agree", "Accept", "Reject all"):
        try:
            btn = page.get_by_role("button", name=label).first
            if btn.is_visible(timeout=800):
                btn.click()
                pause(0.5, 1)
                return
        except Exception:
            continue


def make_row(chain: str, branch: str, source: str, text: str, rating, review_date: date) -> dict:
    return {"restaurant": chain, "branch": branch, "source": source, "review": " ".join(text.split()),
            "rating": rating, "date": review_date.isoformat()}


# ----------------------------------------------------------------------------- Google Maps
GOOGLE_CARD_JS = """
() => [...document.querySelectorAll('div.jftiEf')].map(c => {
  const more = c.querySelector('button.w8nwRe'); if (more) { try { more.click(); } catch (e) {} }
  const q = s => { const e = c.querySelector(s); return e ? e.innerText : ''; };
  const r = c.querySelector('span.kvMYJc');
  return { id: c.getAttribute('data-review-id') || '', text: q('span.wiI7pd'), date: q('span.rsqaWe'),
           rating: r ? (r.getAttribute('aria-label') || '') : '' };
})
"""


def google_place_links(page, query: str, keywords: list[str], max_places: int) -> list[str]:
    page.goto("https://www.google.com/maps/search/" + query.replace(" ", "+"), timeout=60000)
    pause(4, 6)
    accept_consent(page)
    if "/maps/place/" in page.url:
        return [page.url]
    feed = page.locator('div[role="feed"]')
    for _ in range(4):
        try:
            feed.evaluate("e => e.scrollBy(0, 4000)")
        except Exception:
            break
        pause(1, 2)
    links = []
    for a in page.locator("a.hfpxzc").all():
        name = (a.get_attribute("aria-label") or "").lower()
        href = a.get_attribute("href")
        if href and any(k in name for k in keywords) and href not in links:
            links.append(href)
    return links[:max_places]


def scrape_google_place(page, url: str, chain: str, cutoff: date, max_reviews: int) -> list[dict]:
    page.goto(url, timeout=60000)
    pause(4, 6)
    accept_consent(page)
    try:
        branch = page.locator("h1").first.inner_text(timeout=5000).strip()
    except Exception:
        branch = "Unknown"
    try:
        page.get_by_role("tab", name=re.compile("reviews", re.I)).first.click(timeout=8000)
        pause(2, 3)
        page.get_by_role("button", name=re.compile("sort", re.I)).first.click(timeout=5000)
        pause(1, 1.5)
        page.get_by_role("menuitemradio", name=re.compile("newest", re.I)).first.click(timeout=5000)
        pause(2, 3)
    except Exception:
        log.warning("Google: could not open/sort reviews for %s (results may not be newest-first)", branch)

    today = date.today()
    rows: dict[str, dict] = {}
    stagnant, last_count = 0, -1
    scrolls = 0
    while stagnant < 4 and len(rows) < max_reviews and scrolls < 200:
        scrolls += 1
        cards = page.evaluate(GOOGLE_CARD_JS)
        oldest = None
        for c in cards:
            d = parse_date(c["date"], today)
            if d is None:
                continue
            oldest = d if oldest is None else min(oldest, d)
            if cutoff <= d <= WINDOW_END and c["text"].strip():
                key = c["id"] or f'{d}|{c["text"][:60]}'
                rows[key] = make_row(chain, branch, "Google", c["text"], parse_rating(c["rating"]), d)
        if oldest is not None and oldest < cutoff:
            break
        stagnant = stagnant + 1 if len(cards) == last_count else 0
        last_count = len(cards)
        try:
            page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf").first.evaluate("e => e.scrollTo(0, e.scrollHeight)")
        except Exception:
            page.mouse.wheel(0, 4000)
        pause(1.5, 2.5)
    log.info("Google | %s | %s -> %d reviews", chain, branch, len(rows))
    return list(rows.values())


def scrape_google(page, chain: str, queries: list[str], cutoff: date, max_reviews: int) -> list[dict]:
    rows, seen = [], set()
    for q in queries:
        try:
            for link in google_place_links(page, q, CHAIN_KEYWORDS[chain], max_places=8):
                key = link.split("?")[0]
                if key in seen:
                    continue
                seen.add(key)
                try:
                    rows += scrape_google_place(page, link, chain, cutoff, max_reviews)
                except Exception as exc:
                    log.error("Google place failed (%s): %s", link[:80], exc)
        except Exception as exc:
            log.error("Google query failed (%s): %s", q, exc)
    return rows


# ----------------------------------------------------------------------------- Zomato
ZOMATO_DATE_RE = re.compile(
    r"^(\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago|an?\s+\w+\s+ago|yesterday|today|"
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2})$", re.I)


def parse_zomato_text(text: str, today: date) -> list[tuple[str, float | None, date]]:
    """Reads the visible page text: <rating> / DINING|DELIVERY / <date> / <review body>."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = []
    for i, line in enumerate(lines):
        if line.upper() not in ("DINING", "DELIVERY") or i + 1 >= len(lines):
            continue
        if not ZOMATO_DATE_RE.match(lines[i + 1]):
            continue
        d = parse_date(lines[i + 1], today)
        rating = None
        for back in range(1, 5):
            if i - back >= 0 and re.fullmatch(r"[1-5](?:\.\d)?", lines[i - back]):
                rating = float(lines[i - back])
                break
        body = []
        for ln in lines[i + 2:]:
            low = ln.lower()
            if "votes for helpful" in low or low in ("helpful", "comments") or low.upper() in ("DINING", "DELIVERY"):
                break
            if low in ("read more", "follow") or len(ln) <= 3:
                continue
            body.append(ln)
        if d and body:
            out.append((" ".join(body), rating, d))
    return out


def scrape_zomato_page(page, url: str, chain: str, branch: str, cutoff: date, max_scrolls: int = 150) -> list[dict]:
    base = url.rstrip("/")
    page.goto(base if base.endswith("/reviews") else base + "/reviews", timeout=60000)
    pause(4, 6)
    accept_consent(page)
    try:
        page.get_by_text(re.compile(r"^Newest First$", re.I)).first.click(timeout=4000)
        pause(2, 3)
    except Exception:
        log.warning("Zomato: could not set 'Newest First' for %s", branch)
    today = date.today()
    rows: dict[str, dict] = {}
    stagnant = 0
    for _ in range(max_scrolls):
        parsed = parse_zomato_text(page.inner_text("body"), today)
        before = len(rows)
        for body, rating, d in parsed:
            if cutoff <= d <= WINDOW_END:
                rows[f"{d}|{body[:60]}"] = make_row(chain, branch, "Zomato", body, rating, d)
        if parsed and min(d for _, _, d in parsed) < cutoff:
            break
        stagnant = stagnant + 1 if len(rows) == before else 0
        if stagnant >= 4:
            break
        try:
            page.get_by_text(re.compile(r"^(Load more|Show more|Read more reviews)$", re.I)).first.click(timeout=1000)
        except Exception:
            pass
        page.mouse.wheel(0, 5000)
        pause(1.5, 2.5)
    log.info("Zomato | %s | %s -> %d reviews", chain, branch, len(rows))
    return list(rows.values())


def scrape_zomato(page, chain: str, targets: list[dict], cutoff: date) -> list[dict]:
    rows = []
    for t in targets:
        try:
            rows += scrape_zomato_page(page, t["url"], chain, t["branch"], cutoff)
        except Exception as exc:
            log.error("Zomato failed (%s): %s", t["url"], exc)
    return rows


# ----------------------------------------------------------------------------- TripAdvisor
TA_CARD_JS = """
() => [...document.querySelectorAll('[data-automation="reviewCard"], div[data-test-target="HR_CC_CARD"]')].map(c => {
  const more = c.querySelector('button, span'); 
  let rating = '';
  const svg = c.querySelector('svg[aria-label*="bubble"], title');
  if (svg) rating = svg.getAttribute('aria-label') || svg.textContent || '';
  if (!rating) { const m = c.innerText.match(/(\\d(?:\\.\\d)?)\\s+of\\s+5\\s+bubbles/i); if (m) rating = m[0]; }
  if (!rating) { const b = c.querySelector('[class*="bubble_"]'); if (b) { const m = b.className.match(/bubble_(\\d+)/); if (m) rating = m[1]; } }
  return { text: c.innerText, rating };
})
"""
TA_DATE_RES = [re.compile(r"Written\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", re.I),
               re.compile(r"Date of visit:\s*([A-Za-z]+\s+\d{4})", re.I),
               re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b"),
               re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4})\b")]
TA_SKIP = ("read more", "helpful", "share", "report", "contribution", "written", "date of visit", "this review")


def parse_ta_card(card: dict, today: date) -> tuple[str, float | None, date] | None:
    d = None
    for rx in TA_DATE_RES:
        m = rx.search(card["text"])
        if m:
            d = parse_date(m.group(1), today)
            break
    lines = [ln.strip() for ln in card["text"].splitlines() if len(ln.strip()) >= 20]
    lines = [ln for ln in lines if not ln.lower().startswith(TA_SKIP)]
    if d is None or not lines:
        return None
    return max(lines, key=len), parse_rating(card["rating"]), d


def ta_page_url(url: str, page_no: int) -> str:
    if page_no == 0:
        return url
    return re.sub(r"-Reviews-", f"-Reviews-or{page_no * 10}-", url, count=1)


def scrape_tripadvisor_page(page, url: str, chain: str, branch: str, cutoff: date, max_pages: int = 40) -> list[dict]:
    today = date.today()
    rows: dict[str, dict] = {}
    empty_pages = 0
    for n in range(max_pages):
        page.goto(ta_page_url(url, n), timeout=60000)
        pause(3, 5)
        if n == 0:
            accept_consent(page)
        cards = page.evaluate(TA_CARD_JS)
        if not cards:
            break
        oldest = None
        for card in cards:
            parsed = parse_ta_card(card, today)
            if not parsed:
                continue
            body, rating, d = parsed
            oldest = d if oldest is None else min(oldest, d)
            if cutoff <= d <= WINDOW_END:
                rows[f"{d}|{body[:60]}"] = make_row(chain, branch, "TripAdvisor", body, rating, d)
        if oldest is None:
            empty_pages += 1
            if empty_pages >= 2:
                break
        elif oldest < cutoff:
            break
    log.info("TripAdvisor | %s | %s -> %d reviews", chain, branch, len(rows))
    return list(rows.values())


def scrape_tripadvisor(page, chain: str, targets: list[dict], cutoff: date) -> list[dict]:
    rows = []
    for t in targets:
        try:
            rows += scrape_tripadvisor_page(page, t["url"], chain, t["branch"], cutoff)
        except Exception as exc:
            log.error("TripAdvisor failed (%s): %s", t["url"], exc)
    return rows


# ----------------------------------------------------------------------------- runner
def save(rows: list[dict], source: str, chain: str, run_day: date) -> int:
    if not rows:
        log.warning("%s | %s: nothing collected, previous file (if any) kept", source, chain)
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(subset=["branch", "review", "date"])
    name = re.sub(r"[^A-Za-z0-9]+", "_", chain).strip("_") + ".csv"
    for folder in (OUT_DIR / source, OUT_DIR / "_archive" / run_day.isoformat() / source):
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / name, index=False, encoding="utf-8-sig")
    return len(df)


def run(sources: list[str], chains: list[str] | None, months: int, headless: bool, max_reviews: int,
        start: date | None = None, end: date | None = None) -> int:
    from playwright.sync_api import sync_playwright

    global WINDOW_END
    WINDOW_END = end or date.max

    targets = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))["chains"]
    cutoff = start or date.today() - timedelta(days=round(months * 30.4))
    log.info("Collecting reviews dated %s to %s from %s", cutoff, WINDOW_END, ", ".join(sources))
    total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(locale="en-IN", viewport={"width": 1366, "height": 900},
                                      user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        for source in sources:
            for chain, cfg in targets.items():
                if chains and chain not in chains:
                    continue
                page = context.new_page()
                try:
                    if source == "google":
                        rows = scrape_google(page, chain, cfg.get("google", []), cutoff, max_reviews)
                    elif source == "zomato":
                        rows = scrape_zomato(page, chain, cfg.get("zomato", []), cutoff)
                    else:
                        rows = scrape_tripadvisor(page, chain, cfg.get("tripadvisor", []), cutoff)
                except Exception as exc:
                    log.error("%s | %s failed: %s", source, chain, exc)
                    rows = []
                finally:
                    page.close()
                total += save(rows, SOURCES[source], chain, date.today())
        browser.close()
    log.info("Scraping finished: %d reviews written to %s", total, OUT_DIR)
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    ap.add_argument("--chains", nargs="+", help="chain names exactly as in targets.json (default: all)")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--start", type=date.fromisoformat, help="oldest review date, YYYY-MM-DD (overrides --months)")
    ap.add_argument("--end", type=date.fromisoformat, help="newest review date, YYYY-MM-DD (default: today)")
    ap.add_argument("--max-reviews", type=int, default=400, help="cap per Google place")
    ap.add_argument("--headful", action="store_true", help="show the browser window (useful for captchas)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return 0 if run(args.sources, args.chains, args.months, not args.headful, args.max_reviews,
               args.start, args.end) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
