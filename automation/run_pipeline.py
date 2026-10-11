"""End-to-end pipeline: scrape -> merge -> clean -> embeddings -> UMAP -> HDBSCAN -> samples -> labels
-> classify -> issues_by_restaurant.csv -> load into the Streamlit app database.

Usage:
    python automation/run_pipeline.py                      # full run, last 6 months
    python automation/run_pipeline.py --skip-scrape        # reuse the files already in raw_reviews_auto
    python automation/run_pipeline.py --from-step merge    # resume from a step
    python automation/run_pipeline.py --only scrape --sources google --headful
    python automation/run_pipeline.py --list
    python automation/run_pipeline.py --schedule 02:00     # keep running, repeat daily at 02:00
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts_new792026"
RAW_AUTO = ROOT / "Analysis" / "data" / "raw_reviews_auto"
LOG_DIR = ROOT / "logs"

STEPS = ["scrape", "merge", "clean", "embeddings", "umap", "hdbscan", "samples", "label", "classify",
         "issues_csv", "load_app"]
SCRIPT_FOR = {
    "merge": "merge_reviews.py",
    "clean": "clean_reviews.py",
    "embeddings": "generate_embeddings.py",
    "umap": "Reduced_dimention.py",
    "hdbscan": "run_hdbscan.py",
    "samples": "Cluster_representative_samples.py",
    "label": "Autolabelling.py",
    "classify": "apply_classifier.py",
}
log = logging.getLogger("pipeline")


def child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["REVIEWS_INPUT_DIR"] = str(RAW_AUTO)
    return env


def run_cmd(cmd: list[str], cwd: Path) -> None:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=child_env(), text=True, encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout.splitlines()[-40:]:
        log.info("    %s", line)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def step_scrape(args) -> None:
    cmd = [sys.executable, str(ROOT / "automation" / "scrape_reviews.py"), "--months", str(args.months),
           "--sources", *args.sources]
    if args.chains:
        cmd += ["--chains", *args.chains]
    if args.headful:
        cmd.append("--headful")
    run_cmd(cmd, ROOT)


def step_issues_csv(args) -> None:
    import pandas as pd

    df = pd.read_excel(ROOT / "outputs" / "reviews_with_issue_classification.xlsx")
    col = "predicted_issue_category"
    out = (df.groupby(["restaurant", col]).size().reset_index(name="Frequency")
             .rename(columns={col: "Issue"}))
    out.to_csv(ROOT / "issues_by_restaurant.csv", index=False)
    log.info("issues_by_restaurant.csv written (%d rows)", len(out))


def app_dir() -> Path:
    for p in (ROOT, ROOT / "empathy_engine", SCRIPTS / "empathy_engine (5)" / "empathy_engine"):
        if (p / "engine").is_dir():
            return p
    raise FileNotFoundError("engine package not found")


def step_load_app(args) -> None:
    """Copy the scraped files into the app's raw/<source> folders and import them into its database."""
    adir = app_dir()
    for src_folder in ("Google", "Zomato", "TripAdvisor"):
        src = RAW_AUTO / src_folder
        if not src.is_dir():
            continue
        dest = adir / "data" / "raw" / src_folder.lower()
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.csv"):
            shutil.copy2(f, dest / f"auto_{f.name}")
    code = ("from datetime import date; from engine.importer import import_folder; "
            "r = import_folder(ref=date.today()); "
            "print('imported', sum(x['added'] for x in r), 'new reviews from', len(r), 'files')")
    run_cmd([sys.executable, "-c", code], adir)


def run_step(name: str, args) -> None:
    if name == "scrape":
        step_scrape(args)
    elif name == "issues_csv":
        step_issues_csv(args)
    elif name == "load_app":
        step_load_app(args)
    else:
        run_cmd([sys.executable, str(SCRIPTS / SCRIPT_FOR[name])], ROOT)


def select_steps(args) -> list[str]:
    if args.only:
        return args.only
    steps = STEPS[STEPS.index(args.from_step):]
    if args.skip_scrape and "scrape" in steps:
        steps.remove("scrape")
    return [s for s in steps if s not in (args.skip or [])]


def run_pipeline(args) -> int:
    steps = select_steps(args)
    log.info("Pipeline started: %s", " -> ".join(steps))
    for name in steps:
        started = time.time()
        log.info("=== STEP %s ===", name)
        try:
            run_step(name, args)
        except Exception as exc:
            log.error("Step '%s' failed: %s", name, exc)
            if not args.keep_going:
                log.error("Stopping. Fix the problem and resume with: --from-step %s", name)
                return 1
        log.info("Step %s done in %.0fs", name, time.time() - started)
    log.info("Pipeline finished")
    return 0


def seconds_until(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    now = datetime.now()
    nxt = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if nxt <= now:
        nxt = nxt + timedelta(days=1)
    return (nxt - now).total_seconds()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--sources", nargs="+", default=["google", "zomato", "tripadvisor"])
    ap.add_argument("--chains", nargs="+")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--skip-scrape", action="store_true")
    ap.add_argument("--from-step", choices=STEPS, default=STEPS[0])
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--skip", nargs="+", choices=STEPS)
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    ap.add_argument("--schedule", metavar="HH:MM", help="run daily at this local time (keeps running)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(STEPS))
        return 0
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"pipeline_{datetime.now():%Y%m%d}.log",
                                                      encoding="utf-8")])
    if not args.schedule:
        return run_pipeline(args)
    while True:
        wait = seconds_until(args.schedule)
        log.info("Next run in %.1f h", wait / 3600)
        time.sleep(wait)
        run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
