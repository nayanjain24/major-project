"""WLASL dataset processor for VERS sign language model.

Downloads and processes the Word-Level American Sign Language (WLASL) dataset,
extracting MediaPipe hand landmarks from videos and mapping relevant signs
to the VERS emergency vocabulary.

WLASL contains 2,000 ASL words with ~21,000 videos.
We only extract the subset matching our emergency vocabulary for transfer learning.

Usage::
    python -m src.training.wlasl_processor

References:
    - WLASL paper: https://arxiv.org/abs/1910.11006
    - Dataset: https://dxli94.github.io/WLASL/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.vision.sequence_buffer import SequenceBuffer
from src.vision.sign_language_model import SIGN_VOCABULARY

logger = logging.getLogger("vers.training.wlasl")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
WLASL_DIR = DATA_DIR / "wlasl"
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
OUTPUT_DIR = DATA_DIR / "sign_sequences"
SEQUENCE_LENGTH = 30

# Map WLASL glosses → our emergency vocabulary
# WLASL uses lowercase glosses; we map to our uppercase labels
WLASL_TO_VERS: dict[str, str] = {
    "help":       "HELP",
    "stop":       "STOP",
    "accident":   "ACCIDENT",
    "medical":    "MEDICAL",
    "medicine":   "MEDICAL",
    "fire":       "FIRE",
    "police":     "POLICE",
    "ambulance":  "AMBULANCE",
    "danger":     "DANGER",
    "dangerous":  "DANGER",
    "pain":       "PAIN",
    "hurt":       "PAIN",
    "fall":       "FALL",
    "safe":       "SAFE",
    "yes":        "YES",
    "no":         "NO",
    "please":     "PLEASE",
    "emergency":  "EMERGENCY",
}


def download_wlasl_json() -> Optional[Path]:
    """Download the WLASL annotation JSON file."""
    WLASL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = WLASL_DIR / "WLASL_v0.3.json"

    if json_path.exists():
        logger.info("WLASL JSON already exists at %s", json_path)
        return json_path

    logger.info("Downloading WLASL annotations...")
    try:
        urllib.request.urlretrieve(WLASL_JSON_URL, str(json_path))
        logger.info("Downloaded WLASL JSON to %s", json_path)
        return json_path
    except Exception as exc:
        logger.error("Failed to download WLASL JSON: %s", exc)
        return None


def extract_landmarks_from_video(video_path: str) -> Optional[np.ndarray]:
    """Extract a 30-frame landmark sequence from a single video.

    Returns shape (30, 63) or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    buf = SequenceBuffer(window_size=SEQUENCE_LENGTH)
    max_frames = 300  # Safety limit

    frame_count = 0
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            vec = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32).flatten()
            buf.push(vec)

        if buf.ready:
            break

    cap.release()
    hands.close()

    if buf.ready:
        return buf.get_sequence()
    return None


def process_wlasl_videos(max_per_word: int = 50) -> dict[str, int]:
    """Process downloaded WLASL videos and extract landmark sequences.

    Requires videos to be manually downloaded to data/wlasl/videos/.
    Returns a dict of {word: count} of successfully processed sequences.
    """
    json_path = download_wlasl_json()
    if not json_path:
        return {}

    video_dir = WLASL_DIR / "videos"
    if not video_dir.exists():
        logger.warning(
            "WLASL video directory not found at %s. "
            "Download videos from https://dxli94.github.io/WLASL/ "
            "and place them in this directory.",
            video_dir,
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        return {}

    with open(json_path) as f:
        wlasl_data = json.load(f)

    counts: dict[str, int] = {}

    for entry in wlasl_data:
        gloss = entry.get("gloss", "").lower()
        vers_label = WLASL_TO_VERS.get(gloss)
        if not vers_label:
            continue  # Not in our emergency vocabulary

        out_dir = OUTPUT_DIR / vers_label
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("wlasl_*.npy")))

        if existing >= max_per_word:
            counts[vers_label] = existing
            continue

        instances = entry.get("instances", [])
        processed = existing

        for inst in instances:
            if processed >= max_per_word:
                break

            video_id = inst.get("video_id", "")
            video_file = video_dir / f"{video_id}.mp4"
            if not video_file.exists():
                continue

            seq = extract_landmarks_from_video(str(video_file))
            if seq is not None:
                save_path = out_dir / f"wlasl_{processed:04d}.npy"
                np.save(save_path, seq)
                processed += 1

        counts[vers_label] = processed
        if processed > existing:
            logger.info("WLASL: %s → %d sequences extracted", vers_label, processed)

    return counts


def generate_wlasl_synthetic(samples_per_word: int = 60) -> None:
    """Generate WLASL-style augmented synthetic data.

    Applies additional augmentation transforms to simulate the variety
    found in the WLASL dataset (different signers, camera angles,
    hand sizes, signing speeds).
    """
    logger.info("Generating WLASL-augmented synthetic data...")

    # Import the base poses from our synthetic generator
    from src.training.generate_synthetic_data import UNIQUE_POSES, _normalise

    for word, base_pose in UNIQUE_POSES.items():
        out_dir = OUTPUT_DIR / word
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("wlasl_*.npy")))

        for i in range(samples_per_word):
            frames = []
            # Random augmentation per sequence
            scale_factor = np.random.uniform(0.7, 1.3)
            rotation_angle = np.random.uniform(-0.3, 0.3)  # radians
            speed_factor = np.random.uniform(0.8, 1.2)
            hand_size_jitter = np.random.uniform(0.85, 1.15)

            cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
            rot_matrix = np.array([
                [cos_r, -sin_r, 0],
                [sin_r,  cos_r, 0],
                [0,      0,     1],
            ], dtype=np.float32)

            for t in range(SEQUENCE_LENGTH):
                frame = base_pose.copy()
                # Scale variation (different hand sizes)
                frame *= hand_size_jitter
                # Rotation augmentation (wrist rotation)
                frame = frame @ rot_matrix.T
                # Scale (distance from camera)
                frame *= scale_factor
                # Spatial noise (different signers)
                frame += np.random.randn(21, 3).astype(np.float32) * 0.03
                # Temporal variation (signing speed)
                t_adj = t * speed_factor
                sway = np.sin(2 * np.pi * t_adj / SEQUENCE_LENGTH) * 0.015
                frame[:, 1] += sway

                frames.append(_normalise(frame))

            seq = np.array(frames, dtype=np.float32)
            save_path = out_dir / f"wlasl_{existing + i:04d}.npy"
            np.save(save_path, seq)

        logger.info("  %s: +%d WLASL-augmented sequences", word, samples_per_word)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("WLASL Dataset Processor for VERS")
    print("=" * 60)

    # Step 1: Try to process real WLASL videos
    counts = process_wlasl_videos()
    if counts:
        print("\nReal WLASL data processed:")
        for word, count in sorted(counts.items()):
            print(f"  {word}: {count} sequences")

    # Step 2: Generate WLASL-style augmented synthetic data
    print("\nGenerating WLASL-augmented synthetic data...")
    generate_wlasl_synthetic(samples_per_word=60)
    print("✅ WLASL processing complete.")
