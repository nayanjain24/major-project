import argparse
import csv
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "data" / "landmarks.csv"
NUM_LANDMARKS = 21


def build_pose(kind):
    wrist = np.array([[0.50, 0.92, 0.00]])
    thumb_straight = np.array([[0.43, 0.82, -0.01], [0.41, 0.70, -0.02], [0.40, 0.55, -0.03], [0.38, 0.40, -0.04]])
    thumb_folded = np.array([[0.43, 0.82, -0.01], [0.45, 0.77, -0.02], [0.55, 0.75, -0.03], [0.65, 0.75, -0.04]])
    thumb_open = np.array([[0.43, 0.82, -0.01], [0.37, 0.74, -0.02], [0.31, 0.67, -0.02], [0.24, 0.61, -0.03]])

    idx_str = [[0.41, 0.77, -0.01], [0.40, 0.63, -0.02], [0.39, 0.48, -0.03], [0.38, 0.32, -0.04]]
    mid_str = [[0.50, 0.75, -0.01], [0.50, 0.58, -0.02], [0.50, 0.42, -0.03], [0.50, 0.26, -0.04]]
    rng_str = [[0.59, 0.78, -0.01], [0.60, 0.63, -0.02], [0.61, 0.49, -0.03], [0.62, 0.35, -0.04]]
    pnk_str = [[0.68, 0.82, 0.00], [0.70, 0.70, -0.01], [0.72, 0.58, -0.02], [0.74, 0.47, -0.03]]

    idx_fld = [[0.41, 0.77, -0.01], [0.41, 0.70, -0.02], [0.41, 0.74, -0.01], [0.41, 0.80, 0.00]]
    mid_fld = [[0.50, 0.75, -0.01], [0.50, 0.68, -0.02], [0.50, 0.72, -0.01], [0.50, 0.78, 0.00]]
    rng_fld = [[0.59, 0.78, -0.01], [0.59, 0.71, -0.02], [0.59, 0.75, -0.01], [0.59, 0.81, 0.00]]
    pnk_fld = [[0.68, 0.82, 0.00], [0.68, 0.75, -0.01], [0.68, 0.79, 0.00], [0.68, 0.85, 0.01]]

    if kind == "EMERGENCY":
        thumb = thumb_open
        fingers = [idx_str, mid_str, rng_str, pnk_str]
    elif kind == "ACCIDENT":
        thumb = thumb_folded
        fingers = [idx_fld, mid_fld, rng_fld, pnk_fld]
    elif kind == "SOS":
        thumb = thumb_folded
        fingers = [idx_str, mid_str, rng_fld, pnk_fld]
    elif kind == "SAFETY":
        thumb = thumb_straight
        fingers = [idx_fld, mid_fld, rng_fld, pnk_fld]
    else:
        raise ValueError(f"Unknown pose kind: {kind}")

    return np.vstack([wrist, thumb, *[np.array(finger) for finger in fingers]])


def transform_pose(pose, rng):
    centered = pose.copy()
    centered[:, :2] -= centered[0, :2]

    angle = rng.normal(0.0, 0.10)
    scale = rng.uniform(0.90, 1.10)
    tx = rng.uniform(0.42, 0.58)
    ty = rng.uniform(0.84, 0.95)

    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    centered[:, :2] = centered[:, :2] @ rot.T
    centered[:, :2] *= scale
    centered[:, 0] += tx
    centered[:, 1] += ty
    centered[:, 2] += rng.normal(0.0, 0.01, size=centered.shape[0])
    centered += rng.normal(0.0, 0.008, size=centered.shape)
    centered[:, 0] = np.clip(centered[:, 0], 0.0, 1.0)
    centered[:, 1] = np.clip(centered[:, 1], 0.0, 1.0)
    return centered


def main():
    parser = argparse.ArgumentParser(description="Generate demo landmarks for ACCIDENT, EMERGENCY, SOS, and SAFETY.")
    parser.add_argument("--samples-per-class", type=int, default=180)
    parser.add_argument("--outfile", default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    labels = ["ACCIDENT", "EMERGENCY", "SOS", "SAFETY"]
    bases = {label: build_pose(label) for label in labels}

    with outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f_{i}" for i in range(NUM_LANDMARKS * 3)])
        for label in labels:
            for _ in range(args.samples_per_class):
                pose = transform_pose(bases[label], rng)
                writer.writerow([label, *pose.reshape(-1).tolist()])

    print(f"Generated demo dataset: {outfile}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Classes: {', '.join(labels)}")


if __name__ == "__main__":
    main()
