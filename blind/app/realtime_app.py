"""Real-time application loop.

This module orchestrates:
1. Camera capture via OpenCV.
2. Landmark extraction via MediaPipe.
3. Feature preprocessing + PyTorch model inference.
4. Rendering/printing predictions.
"""

from __future__ import annotations

import cv2
import torch

from models.sign_language_model import SignLanguageClassifier
from utils.mediapipe_tracker import MediaPipeTracker
from utils.preprocessing import landmarks_to_tensor


def run_realtime_translation() -> None:
    """Start the real-time camera loop for sign-language translation."""
    tracker = MediaPipeTracker()
    model = SignLanguageClassifier()
    model.eval()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        tracker.close()
        return

    print("Running real-time translation. Press 'q' to quit.")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to read frame.")
                break

            landmarks = tracker.process_frame(frame)
            features = landmarks_to_tensor(landmarks).unsqueeze(0)
            logits = model(features)
            predicted_class = int(torch.argmax(logits, dim=1).item())

            cv2.putText(
                frame,
                f"Predicted Class: {predicted_class}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Sign Language Translation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
