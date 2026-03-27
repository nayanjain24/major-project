"""Train the Phase-1 gesture classifier from extracted hand landmarks.

Phase-1 Alignment
-----------------
- Methodology #4: Supervised learning model for gesture recognition.
- Methodology #5: Model evaluation before real-time deployment.
- Objectives 1-5: Core AI/ML module for live emergency communication.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - optional dependency
    class Console:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)

    console = Console()
    Table = None  # type: ignore[assignment]
else:
    console = Console()

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_utils import DATA_PATH, MODEL_PATH, ensure_project_dirs


def _render_confusion_matrix(labels: list[str], matrix) -> None:
    if Table is None:
        console.print("Confusion Matrix")
        console.print(",".join(["Actual \\ Pred", *labels]))
        for row_index, row in enumerate(matrix):
            console.print(",".join([labels[row_index], *[str(value) for value in row]]))
        return

    table = Table(title="Confusion Matrix", show_header=True)
    table.add_column("Actual \\ Pred", style="cyan")
    for label in labels:
        table.add_column(label, justify="center")
    for row_index, row in enumerate(matrix):
        table.add_row(labels[row_index], *[str(value) for value in row])
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VERS gesture classifier.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to save the trained model bundle.")
    args = parser.parse_args()

    ensure_project_dirs()
    warnings.filterwarnings("ignore", category=UserWarning)

    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        console.print(f"[bold red]Dataset missing at {DATA_PATH}.[/bold red]")
        console.print("Run `python src/record_gestures.py` first to collect landmarks.")
        return

    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    feature_columns = [column for column in df.columns if column.startswith("f_")]
    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 landmark features, found {len(feature_columns)}. "
            "Please regenerate the dataset with `python src/record_gestures.py`."
        )

    if df["label"].nunique() < 2:
        raise ValueError("Need data for at least 2 distinct gestures to train.")

    X = df[feature_columns]
    y = df["label"].astype(str)

    console.print(f"\n[bold green]Loaded {len(df)} samples across {y.nunique()} labels.[/bold green]")
    console.print(f"Class distribution: {dict(Counter(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )

    console.print("\n[bold green]Training gesture classifier...[/bold green]")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = list(pipeline.classes_)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    if Table is not None:
        summary = Table(title="Model Evaluation Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="magenta")
        summary.add_row("Accuracy", f"{accuracy:.3f}")
        summary.add_row("Classes", ", ".join(labels))
        console.print(summary)
    else:
        console.print(f"Accuracy: {accuracy:.3f}")

    console.print("\n[bold]Classification Report[/bold]")
    console.print(classification_report(y_test, y_pred))
    _render_confusion_matrix(labels, matrix)

    model_out = Path(args.model)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "sklearn_pipeline",
            "pipeline": pipeline,
            "labels": labels,
            "feature_columns": feature_columns,
        },
        model_out,
    )
    console.print(f"[cyan]Model saved to {model_out}[/cyan]")


if __name__ == "__main__":
    main()
