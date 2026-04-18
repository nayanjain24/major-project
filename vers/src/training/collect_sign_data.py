"""Sign language data collection utility for VERS v4.0.

Records MediaPipe hand landmark sequences for each emergency vocabulary word.
Each recording captures 30 frames (≈1 second) and saves the normalised
landmarks as a NumPy `.npy` file.

Usage::

    python -m src.training.collect_sign_data

Follow the on-screen prompts to record gestures for each word.
Data is saved to ``data/sign_sequences/{WORD}/seq_NNN.npy``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.data_utils import extract_hand_vector
from src.vision.sequence_buffer import SequenceBuffer
from src.vision.sign_language_model import SIGN_VOCABULARY

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sign_sequences"
SEQUENCE_LENGTH = 30
SEQUENCES_PER_WORD = 30  # Record 30 samples per word


def collect_data() -> None:
    """Interactive data collection loop."""
    print("=" * 60)
    print("VERS Sign Language Data Collection")
    print("=" * 60)
    print(f"\nVocabulary: {', '.join(SIGN_VOCABULARY[1:])}")  # Skip NONE
    print(f"Sequences per word: {SEQUENCES_PER_WORD}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"Output directory: {DATA_DIR}\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    # Skip "NONE" — we don't need to record it (it's the default class)
    words_to_record = [w for w in SIGN_VOCABULARY if w != "NONE"]

    try:
        for word in words_to_record:
            word_dir = DATA_DIR / word
            word_dir.mkdir(parents=True, exist_ok=True)

            # Count existing sequences to resume from where we left off
            existing = len(list(word_dir.glob("*.npy")))
            if existing >= SEQUENCES_PER_WORD:
                print(f"  ✅ {word}: Already have {existing} sequences. Skipping.")
                continue

            print(f"\n{'=' * 40}")
            print(f"  Recording: {word}")
            print(f"  ({existing}/{SEQUENCES_PER_WORD} already collected)")
            print(f"{'=' * 40}")

            for seq_idx in range(existing, SEQUENCES_PER_WORD):
                # Wait for user to press SPACE to start recording
                print(f"\n  [{seq_idx + 1}/{SEQUENCES_PER_WORD}] "
                      f"Show the sign for '{word}' and press SPACE to record...")

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    # Draw hand landmarks for visual feedback
                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS
                            )

                    cv2.putText(
                        frame,
                        f"Sign: {word} | Press SPACE to record | Q to quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )
                    cv2.imshow("VERS Sign Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        break
                    if key == ord("q"):
                        print("\nCollection cancelled by user.")
                        return

                # Record 30 frames
                buf = SequenceBuffer(window_size=SEQUENCE_LENGTH)
                print(f"    Recording {SEQUENCE_LENGTH} frames...", end="", flush=True)

                while not buf.ready:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    hand_vector = extract_hand_vector(results)

                    if hand_vector is not None:
                        buf.push(hand_vector)

                    # Visual feedback: recording indicator
                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS
                            )

                    fill_pct = int(buf.fill_level * 100)
                    cv2.putText(
                        frame,
                        f"RECORDING: {word} [{fill_pct}%]",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    cv2.imshow("VERS Sign Data Collection", frame)
                    cv2.waitKey(1)

                # Save sequence
                sequence = buf.get_sequence()  # (30, 63)
                save_path = word_dir / f"seq_{seq_idx:03d}.npy"
                np.save(save_path, sequence)
                print(f" ✅ Saved to {save_path.name}")

            print(f"\n  ✅ {word}: All {SEQUENCES_PER_WORD} sequences collected.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Data saved to: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    collect_data()
