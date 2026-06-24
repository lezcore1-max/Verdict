"""
VERDICT — Entry point.
Run with:  python run.py   OR   streamlit run frontend/app.py
"""
import sys
import os

# Make verdict/ the import root
sys.path.insert(0, os.path.dirname(__file__))

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "frontend/app.py", "--server.headless", "false"]
    stcli.main()
