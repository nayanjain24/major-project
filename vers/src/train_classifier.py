"""Train the Phase-1 gesture classifier from extracted hand landmarks.

Phase-1 Alignment
-----------------
- Methodology #4: Supervised learning model for gesture recognition.
- Methodology #5: Model evaluation before real-time deployment.
- Objectives 1-5: Core AI/ML module for live emergency communication.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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

from src.utils.data_utils import DATA_PATH, MODEL_DIR, MODEL_PATH, ensure_project_dirs

REPORTS_DIR = MODEL_DIR / "reports"


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


def _save_confusion_matrix_png(labels: list[str], matrix, output_path: Path) -> None:
    """Save a publication-quality confusion matrix as a PNG image."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the count
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, format(matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[cyan]Confusion matrix saved to {output_path}[/cyan]")


def _save_classification_report_json(y_test, y_pred, labels: list[str], output_path: Path) -> None:
    """Save the sklearn classification report as a JSON file."""
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"[cyan]Classification report saved to {output_path}[/cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VERS gesture classifier.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to save the trained model bundle.")
    args = parser.parse_args()

    ensure_project_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        console.print(f"[bold red]Dataset missing at {DATA_PATH}.[/bold red]")
        console.print("Run `python seed_demo_data.py` or `python src/record_gestures.py` first to collect landmarks.")
        return

    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    feature_columns = [column for column in df.columns if column.startswith("f_")]
    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 landmark features, found {len(feature_columns)}. "
            "Please regenerate the dataset with `python seed_demo_data.py`."
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

    console.print("\n[bold green]Augmenting training data for robustness...[/bold green]")
    # Add Gaussian noise to create variants of the original samples
    noise_factor = 0.005
    X_aug = X_train.copy()
    X_aug += np.random.normal(0, noise_factor, X_aug.shape)

    X_final = pd.concat([X_train, X_aug])
    y_final = pd.concat([y_train, y_train])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )

    # Cross-validation on training data before final fit
    console.print("\n[bold green]Running 5-fold cross-validation...[/bold green]")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_final.values, y_final.values, cv=cv, scoring="accuracy")
    console.print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    console.print("\n[bold green]Training gesture classifier...[/bold green]")
    pipeline.fit(X_final.values, y_final.values)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = list(pipeline.classes_)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    if Table is not None:
        summary = Table(title="Model Evaluation Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="magenta")
        summary.add_row("Test Accuracy", f"{accuracy:.3f}")
        summary.add_row("CV Accuracy (mean)", f"{cv_scores.mean():.3f}")
        summary.add_row("CV Accuracy (std)", f"{cv_scores.std():.3f}")
        summary.add_row("Classes", ", ".join(labels))
        summary.add_row("Training Samples", str(len(X_final)))
        summary.add_row("Test Samples", str(len(X_test)))
        console.print(summary)
    else:
        console.print(f"Test Accuracy: {accuracy:.3f}")
        console.print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    console.print("\n[bold]Classification Report[/bold]")
    console.print(classification_report(y_test, y_pred))
    _render_confusion_matrix(labels, matrix)

    # Save reports
    _save_classification_report_json(y_test, y_pred, labels, REPORTS_DIR / "classification_report.json")
    _save_confusion_matrix_png(labels, matrix, REPORTS_DIR / "confusion_matrix.png")

    # Save model bundle with metadata
    model_out = Path(args.model)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type": "sklearn_pipeline",
        "pipeline": pipeline,
        "labels": labels,
        "feature_columns": feature_columns,
        "training_metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(df),
            "training_samples": len(X_final),
            "test_samples": len(X_test),
            "test_accuracy": round(accuracy, 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "augmentation": f"Gaussian noise (σ={noise_factor})",
            "n_estimators": 300,
            "max_depth": 12,
        },
    }
    joblib.dump(bundle, model_out)
    console.print(f"\n[bold cyan]Model saved to {model_out}[/bold cyan]")
    console.print("[bold green]Training complete.[/bold green]")


if __name__ == "__main__":
    main()
