"""Train an LSTM gesture classifier on keypoint sequence .npy files.

Expected dataset layout:
    dataset/processed/
        hello/
            hello_20260101_120000_40f.npy
            hello_20260101_120100_36f.npy
        thanks/
            thanks_20260101_121500_38f.npy

Each .npy file should contain a single sequence with shape [T, D], where:
- T = number of frames in the recorded sequence
- D = keypoint feature dimension (default 258)
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from models import GestureLSTMClassifier


@dataclass
class EpochMetrics:
    """Container for loss/accuracy numbers for one epoch."""

    loss: float
    accuracy: float


class GestureSequenceDataset(Dataset):
    """Loads gesture sequences from folders and normalizes length per sample."""

    def __init__(self, data_dir: Path, seq_len: int, input_dim: int) -> None:
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.input_dim = input_dim

        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {data_dir}")

        gesture_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        if not gesture_dirs:
            raise ValueError(f"No gesture folders found in: {data_dir}")

        self.label_to_index = {gesture_dir.name: i for i, gesture_dir in enumerate(gesture_dirs)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}

        self.samples: List[Tuple[Path, int]] = []
        for gesture_dir in gesture_dirs:
            class_index = self.label_to_index[gesture_dir.name]
            for npy_file in sorted(gesture_dir.glob("*.npy")):
                self.samples.append((npy_file, class_index))

        if not self.samples:
            raise ValueError(f"No .npy files found under gesture folders in: {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path, label = self.samples[idx]
        sequence = np.load(npy_path).astype(np.float32)
        sequence = self._ensure_shape(sequence, npy_path)
        sequence = self._pad_or_truncate(sequence)

        x = torch.from_numpy(sequence)  # [seq_len, input_dim]
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def _ensure_shape(self, sequence: np.ndarray, npy_path: Path) -> np.ndarray:
        """Validate/repair sequence shape so it becomes [T, input_dim]."""
        if sequence.ndim == 1:
            if sequence.shape[0] != self.input_dim:
                raise ValueError(
                    f"{npy_path}: expected 1D vector of {self.input_dim}, got {sequence.shape[0]}"
                )
            sequence = sequence[np.newaxis, :]
        elif sequence.ndim == 2:
            if sequence.shape[1] == self.input_dim:
                pass
            elif sequence.shape[0] == self.input_dim:
                # Handle files stored as [input_dim, T].
                sequence = sequence.T
            else:
                raise ValueError(
                    f"{npy_path}: expected second dim {self.input_dim}, got {sequence.shape}"
                )
        else:
            raise ValueError(f"{npy_path}: expected 1D/2D array, got ndim={sequence.ndim}")

        return sequence

    def _pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """Force a fixed sequence length for batching."""
        frame_count = sequence.shape[0]
        if frame_count == self.seq_len:
            return sequence
        if frame_count > self.seq_len:
            # Keep the latest frames (often most informative in gesture clips).
            return sequence[-self.seq_len :]

        padding = np.zeros((self.seq_len - frame_count, self.input_dim), dtype=np.float32)
        return np.concatenate([sequence, padding], axis=0)


def make_stratified_split(
    labels: Sequence[int], val_split: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Split indices into train/val while trying to keep class balance."""
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, indices in by_label.items():
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        val_count = max(1, int(round(len(indices) * val_split)))
        val_count = min(val_count, len(indices) - 1)  # keep at least 1 train sample per class
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices:
        raise ValueError("Train split is empty. Add more data or reduce val_split.")
    if not val_indices:
        raise ValueError(
            "Validation split is empty. Add more data or use a higher --val-split value."
        )

    return train_indices, val_indices


def choose_device() -> torch.device:
    """Pick the best available device (CUDA, then MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Run one train or validation epoch and return averaged metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # Model returns probabilities. We convert to log-probabilities for NLLLoss.
        probs = model(inputs)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        loss = criterion(torch.log(probs), targets)

        if is_training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (probs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    label_to_index: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    """Persist training state so model can be restored later."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "label_to_index": label_to_index,
        "index_to_label": {idx: label for label, idx in label_to_index.items()},
        "config": vars(args),
    }
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GestureLSTMClassifier on keypoint sequence datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Path with per-gesture folders containing .npy sequence files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where best/last model checkpoints are saved.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--seq-len", type=int, default=30, help="Fixed sequence length for model.")
    parser.add_argument("--input-dim", type=int, default=258, help="Per-frame keypoint vector size.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Model dropout probability.")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use a bidirectional LSTM encoder.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience based on validation loss.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation loss improvement to reset early stopping counter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (set >0 for faster loading on larger datasets).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = GestureSequenceDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
    )

    labels = [label for _, label in dataset.samples]
    train_indices, val_indices = make_stratified_split(
        labels=labels, val_split=args.val_split, seed=args.seed
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device()
    model = GestureLSTMClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(dataset.label_to_index),
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    best_val_loss = float("inf")
    early_stop_counter = 0

    print(f"Device: {device}")
    print(f"Classes ({len(dataset.label_to_index)}): {dataset.label_to_index}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

        # Always save latest checkpoint.
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            label_to_index=dataset.label_to_index,
            args=args,
        )

        # Save best checkpoint + early stopping decision.
        if val_metrics.loss < (best_val_loss - args.min_delta):
            best_val_loss = val_metrics.loss
            early_stop_counter = 0
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                label_to_index=dataset.label_to_index,
                args=args,
            )
            print(f"  New best model saved to: {best_path}")
        else:
            early_stop_counter += 1
            print(f"  No improvement. Early stop counter: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {last_path}")


if __name__ == "__main__":
    main()
