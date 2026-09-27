"""Repository entry point for the Empathy Engine Streamlit app."""
import runpy
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent / "scripts_new792026" / "empathy_engine (5)" / "empathy_engine"
sys.path.insert(0, str(APP_DIR))
runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
