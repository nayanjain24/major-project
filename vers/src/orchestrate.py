"""End-to-end VERS demo orchestrator for a MacBook presentation.

Default behaviour is demo-friendly:
- reuse existing ``data/landmarks.csv`` when present
- reuse existing ``models/gesture_classifier.pkl`` when present

Use ``--force-capture`` and/or ``--force-train`` for a full from-scratch run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from rich import print as rprint
except Exception:  # pragma: no cover - optional dependency
    rprint = print

GESTURES = {"HELP": 180, "MEDICAL": 180, "DANGER": 180}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORD = PROJECT_ROOT / "src" / "record_gestures.py"
TRAIN = PROJECT_ROOT / "src" / "train_classifier.py"
REALTIME = PROJECT_ROOT / "src" / "realtime_vers.py"
ALERT_SERVER = PROJECT_ROOT / "src" / "alert_server.py"
DASHBOARD = PROJECT_ROOT / "src" / "vers_dashboard.py"
DATA_PATH = PROJECT_ROOT / "data" / "landmarks.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_classifier.pkl"


def _run(cmd: list[str], title: str, *, background: bool = False):
    rprint(f"\n[bold magenta]--- {title} ---[/bold magenta]")
    if background:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        rprint(f"[green]{title} started (PID {proc.pid}).[/green]")
        return proc

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        rprint(f"[bold green]{title} completed.[/bold green]")
    except subprocess.CalledProcessError as exc:
        rprint(f"[bold red]{title} failed with exit code {exc.returncode}.[/bold red]")
        raise
    return None


def _venv_active(python_exec: str) -> bool:
    return "/.venv/" in python_exec or python_exec.endswith("/.venv/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser(description="VERS MacBook demo orchestrator.")
    parser.add_argument(
        "--force-capture",
        action="store_true",
        help="Recapture gesture samples even if data/landmarks.csv already exists.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain the classifier even if models/gesture_classifier.pkl already exists.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    python_exec = sys.executable

    if not _venv_active(python_exec):
        rprint("[bold red]Activate the project .venv before running orchestrate.py.[/bold red]")
        sys.exit(1)

    should_capture = args.force_capture or not (DATA_PATH.exists() and DATA_PATH.stat().st_size > 0)
    should_train = args.force_train or not MODEL_PATH.exists()

    if should_capture:
        rprint("[bold blue]Phase 1: Gesture data capture[/bold blue]")
        for label, samples in GESTURES.items():
            _run([python_exec, str(RECORD), "--label", label, "--samples", str(samples)],
                 f"Recording {samples} samples for {label}")
            time.sleep(2)
    else:
        rprint("[cyan]Reusing existing data/landmarks.csv for a fast demo run.[/cyan]")

    if should_train:
        if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
            rprint("[bold red]Cannot train because data/landmarks.csv is missing or empty.[/bold red]")
            sys.exit(1)
        rprint("\n[bold blue]Phase 2: Train gesture classifier[/bold blue]")
        _run([python_exec, str(TRAIN)], "Training model")
    else:
        rprint("[cyan]Reusing existing models/gesture_classifier.pkl for a fast demo run.[/cyan]")

    alert_proc = None
    dashboard_proc = None
    try:
        rprint("\n[bold blue]Phase 3: Launch background services[/bold blue]")
        alert_proc = _run([python_exec, str(ALERT_SERVER)], "Mock alert server", background=True)
        time.sleep(2)

        dashboard_proc = _run(
            [python_exec, "-m", "streamlit", "run", str(DASHBOARD), "--server.headless=true"],
            "Streamlit dashboard",
            background=True,
        )
        rprint("[cyan]Streamlit dashboard available at http://localhost:8501[/cyan]")
        rprint("[cyan]Legacy Flask dashboard remains available via `python web_vers.py`.[/cyan]")
        time.sleep(4)

        rprint("\n[bold blue]Phase 4: Real-time demo[/bold blue]")
        rprint("[cyan]Press 'q' in the OpenCV window to exit.[/cyan]")
        _run([python_exec, str(REALTIME)], "Real-time VERS demo")
    finally:
        rprint("\n[bold blue]Cleaning up background processes...[/bold blue]")
        for proc, name in ((dashboard_proc, "Streamlit dashboard"), (alert_proc, "Alert server")):
            if proc is not None and proc.poll() is None:
                rprint(f"[yellow]Terminating {name} (PID {proc.pid}).[/yellow]")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        rprint("[bold green]All components shut down.[/bold green]")


if __name__ == "__main__":
    main()
