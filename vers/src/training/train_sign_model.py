"""LSTM training script for the VERS sign language classifier.

Loads recorded landmark sequences from ``data/sign_sequences/{WORD}/*.npy``,
trains the SignLanguageLSTM model, and exports weights to
``models/sign_language_lstm.pth``.

Usage::

    python -m src.training.train_sign_model

Requirements:
    - PyTorch (pip install torch)
    - Recorded data from ``collect_sign_data.py``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.vision.sign_language_model import (
    SIGN_VOCABULARY,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    MODEL_PATH,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sign_sequences"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vers.training")


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all recorded sequences and build X, y arrays.

    Returns
    -------
    X : ndarray, shape (N, SEQUENCE_LENGTH, NUM_FEATURES)
    y : ndarray, shape (N,) — integer class indices
    labels : list of class names in index order
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    labels = SIGN_VOCABULARY  # index 0 = NONE, 1 = HELP, etc.

    for class_idx, word in enumerate(labels):
        if word == "NONE":
            continue  # We don't train on NONE — it's the fallback class
        word_dir = DATA_DIR / word
        if not word_dir.exists():
            logger.warning("No data for '%s' — skipping.", word)
            continue

        sequences = sorted(word_dir.glob("*.npy"))
        if not sequences:
            logger.warning("No .npy files in %s — skipping.", word_dir)
            continue

        for seq_path in sequences:
            seq = np.load(seq_path)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X_list.append(seq)
                y_list.append(class_idx)
            else:
                logger.warning("Skipping %s: unexpected shape %s", seq_path.name, seq.shape)

    if not X_list:
        raise FileNotFoundError(
            f"No training data found in {DATA_DIR}. "
            f"Run `python -m src.training.collect_sign_data` first."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    logger.info("Loaded %d sequences across %d classes.", len(X), len(set(y_list)))
    return X, y, labels


def train() -> None:
    """Train the LSTM model and save weights."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch")
        sys.exit(1)

    from src.vision.sign_language_model import SignLanguageLSTM

    # --- Load data ---
    X, y, labels = load_dataset()

    # --- Train/val split (80/20) ---
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    val_ds = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    # --- Model ---
    model = SignLanguageLSTM(num_classes=len(labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 15
    epochs = 100

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

        train_loss /= total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | Train loss: %.4f acc: %.2f%% | Val loss: %.4f acc: %.2f%%",
                epoch, train_loss, train_acc * 100, val_loss, val_acc * 100,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Model saved to: %s", MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
