"""Capture hand-landmark vectors for a single gesture label.

Phase-1 Alignment
-----------------
- Methodology #1: Camera input via OpenCV ``VideoCapture``.
- Methodology #2: Hand detection with MediaPipe Hands.
- Methodology #3: Landmark extraction -> 63-element feature vector.
- Objective 2: Detect and recognise sign-language gestures.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import mediapipe as mp

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from rich import print as rprint
except Exception:  # pragma: no cover - optional dependency
    rprint = print

from src.utils.data_utils import (
    DATA_PATH,
    csv_header,
    ensure_project_dirs,
    extract_hand_vector,
    open_camera_capture,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VERS gesture recorder (Phase-1 Methodology #1-3).",
    )
    parser.add_argument("--label", required=True, help="Gesture label, e.g. HELP, MEDICAL, DANGER.")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to capture.")
    parser.add_argument("--outfile", default=str(DATA_PATH), help="Output CSV path.")
    args = parser.parse_args()

    ensure_project_dirs()

    output_path = Path(args.outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not output_path.exists() or os.path.getsize(output_path) == 0

    cap, _backend_info = open_camera_capture(max_index=4, warmup_reads=18)
    if cap is None:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot access webcam. Check macOS camera permissions "
            "(System Settings -> Privacy & Security -> Camera) and close other camera apps."
        )

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands, output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if need_header:
            writer.writerow(csv_header())

        collected = 0
        cooldown = time.time()

        rprint(f"\n[bold green]Recording {args.samples} samples for {args.label.upper()}[/bold green]")
        rprint("[cyan]Keep one hand visible and centred. Press 'q' to quit early.[/cyan]")

        try:
            while collected < args.samples:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                vector = extract_hand_vector(results)

                now = time.time()
                if vector is not None and now - cooldown > 0.08:
                    writer.writerow([args.label.upper(), *vector.flatten().tolist()])
                    collected += 1
                    cooldown = now

                overlay = frame.copy()
                cv2.putText(
                    overlay,
                    f"Label: {args.label.upper()} ({collected}/{args.samples})",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    "Press 'q' to stop",
                    (12, overlay.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )

                if getattr(results, "multi_hand_landmarks", None):
                    for hand_lms in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            overlay,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                cv2.imshow("VERS Gesture Capture (macOS)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rprint("[yellow]Recording interrupted by user.[/yellow]")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    rprint(f"[bold magenta]Capture complete. {collected} samples stored in {output_path}.[/bold magenta]")


if __name__ == "__main__":
    main()
