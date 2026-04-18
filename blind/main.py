"""Main entry point for the laptop-webcam VERS fallback demo.

This keeps `python main.py` useful even when the Streamlit dashboard is not used.
"""

from app.vers_demo import run_demo


if __name__ == "__main__":
    run_demo()
