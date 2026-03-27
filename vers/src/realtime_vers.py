"""Real-time VERS demo: gesture recognition, distress scoring, and alerting.

Phase-1 Alignment
-----------------
- System Design: Input -> Preprocessing -> Feature Extraction -> AI Module -> Alert -> Communication
- Methodology #1-#7: Full real-time pipeline from camera feed to structured alert output
- Objectives 1-7: Gesture recognition, distress analysis, accessibility, and inclusive response
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path(__file__).resolve().parent.parent / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

try:
    from rich import print as rprint
except Exception:  # pragma: no cover - optional dependency
    rprint = print

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.alert_utils import ALERT_MAP, log_alert, log_error, make_alert_payload
from src.utils.data_utils import (
    DISTRESS_HISTORY_PATH,
    MODEL_PATH,
    ensure_project_dirs,
    extract_hand_vector,
)

ALERT_ENDPOINT = "http://localhost:8000/alert"
DISTRESS_THRESHOLD = 0.055
HAND_CONF_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5
ALERT_COOLDOWN_SECONDS = 5
DISTRESS_HISTORY_LIMIT = 200

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
BROW_LEFT, EYE_LEFT = 70, 159
BROW_RIGHT, EYE_RIGHT = 300, 386


def load_model() -> tuple[dict[str, Any], list[str]]:
    """Load the trained model bundle used by real-time inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model missing at {MODEL_PATH}. Run `python src/train_classifier.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    labels = list(bundle.get("labels", []))
    if "pipeline" in bundle:
        return bundle, labels
    if bundle.get("model_type") == "centroid" and "centroids" in bundle and "scales" in bundle:
        return bundle, labels
    raise ValueError("Unsupported model bundle format.")


def predict_gesture(model_bundle: dict[str, Any], hand_vector: np.ndarray) -> tuple[str, float]:
    """Predict the gesture label and confidence for a single hand vector."""
    if "pipeline" in model_bundle:
        pipeline = model_bundle["pipeline"]
        probs = pipeline.predict_proba(hand_vector)[0]
        best_index = int(np.argmax(probs))
        return str(pipeline.classes_[best_index]), float(probs[best_index])

    if model_bundle.get("model_type") == "centroid":
        vector = hand_vector.reshape(-1).astype(np.float32)
        labels = list(model_bundle["labels"])
        scores: list[tuple[str, float]] = []
        for label in labels:
            centroid = np.asarray(model_bundle["centroids"][label], dtype=np.float32)
            scales = np.asarray(model_bundle["scales"][label], dtype=np.float32)
            scales = np.where(np.abs(scales) < 1e-6, 1.0, scales)
            distance = np.linalg.norm((vector - centroid) / scales)
            scores.append((label, 1.0 / (1.0 + float(distance))))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[0]

    raise ValueError("Unsupported model bundle format during prediction.")


def calc_distress(face_lms: Any, width: int, height: int) -> float:
    """Compute the demo distress heuristic from MediaPipe face landmarks."""
    if face_lms is None:
        return 0.0

    def pt(index: int) -> np.ndarray:
        landmark = face_lms.landmark[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    try:
        mouth_vertical = np.linalg.norm(pt(MOUTH_TOP) - pt(MOUTH_BOTTOM))
        mouth_horizontal = np.linalg.norm(pt(MOUTH_LEFT) - pt(MOUTH_RIGHT))
        brow_left_gap = np.linalg.norm(pt(BROW_LEFT) - pt(EYE_LEFT))
        brow_right_gap = np.linalg.norm(pt(BROW_RIGHT) - pt(EYE_RIGHT))

        mouth_ratio = mouth_vertical / max(mouth_horizontal, 1e-6)
        brow_ratio = (brow_left_gap + brow_right_gap) / (2 * height)
        return float(0.65 * mouth_ratio + 0.35 * brow_ratio)
    except Exception:
        return 0.0


def smooth_prediction(history: deque[tuple[str, float]]) -> tuple[str, float]:
    """Combine recent predictions into a more stable display label."""
    if not history:
        return "NONE", 0.0

    weighted: Counter[str] = Counter()
    confidences: dict[str, list[float]] = {}
    for label, confidence in history:
        weighted[label] += float(confidence)
        confidences.setdefault(label, []).append(float(confidence))

    valid = {label: score for label, score in weighted.items() if label != "NONE"}
    if not valid:
        return "NONE", 0.0

    best_label = max(valid, key=lambda label: valid[label])
    return best_label, float(np.mean(confidences[best_label]))


def append_distress_history(entries: deque[str], distress_score: float, distress_flag: bool) -> None:
    """Persist the most recent distress scores for later analysis."""
    entries.append(f"{datetime.now().isoformat()},{distress_score:.4f},{distress_flag}")
    DISTRESS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISTRESS_HISTORY_PATH.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,distress_score,distress_flag\n")
        handle.write("\n".join(entries))
        handle.write("\n")


def draw_overlay(
    frame: np.ndarray,
    hand_results: Any,
    face_lms: Any,
    display_label: str,
    display_confidence: float,
    distress_score: float,
    distress_flag: bool,
) -> np.ndarray:
    """Render the real-time demo overlay for both OpenCV and dashboard views."""
    overlay = frame.copy()
    shown_label = display_label if display_label != "NONE" else "No gesture"
    shown_conf = display_confidence if display_label != "NONE" else 0.0

    if shown_label in ["ACCIDENT", "EMERGENCY", "SOS"]:
        text_color = (0, 0, 255) # BGR Red
    elif shown_label == "SAFETY":
        text_color = (0, 255, 0) # BGR Green
    else:
        text_color = (255, 255, 0)

    cv2.putText(
        overlay,
        f"Gesture: {shown_label}",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Confidence: {shown_conf:.2f}",
        (12, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Distress Score: {distress_score:.3f}",
        (12, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255) if distress_flag else (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Press 'q' to quit",
        (12, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if getattr(hand_results, "multi_hand_landmarks", None):
        for hand_lms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                overlay,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    if face_lms is not None:
        mp_drawing.draw_landmarks(
            overlay,
            face_lms,
            mp_face.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            overlay,
            face_lms,
            mp_face.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )

    return overlay


def main() -> None:
    ensure_project_dirs()

    rprint("[bold blue]Starting VERS real-time demo[/bold blue]")
    model_bundle, labels = load_model()
    rprint(f"[cyan]Loaded classifier with gestures: {', '.join(labels)}[/cyan]")
    rprint("[cyan]Press 'q' to exit the OpenCV window.[/cyan]")

    cap = None
    for cam_idx in range(3):
        c = cv2.VideoCapture(cam_idx)
        if c.isOpened():
            ok, _ = c.read()
            if ok:
                cap = c
                break
            c.release()
    
    if cap is None:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot access webcam. Check macOS permissions and close other camera apps."
        )

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    recent_preds: deque[tuple[str, float]] = deque(maxlen=SMOOTHING_WINDOW)
    distress_history: deque[str] = deque(maxlen=DISTRESS_HISTORY_LIMIT)
    last_alert_signature = ""
    last_alert_time = 0.0

    try:
        while True:
            try:
                ok, frame = cap.read()
                if not ok:
                    log_error("Frame capture failed; skipping frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape

                hand_results = hands.process(rgb)
                face_results = face_mesh.process(rgb)
                hand_vector = extract_hand_vector(hand_results)

                gesture_label, gesture_confidence = "NONE", 0.0
                if hand_vector is not None:
                    gesture_label, gesture_confidence = predict_gesture(model_bundle, hand_vector)
                recent_preds.append((gesture_label, gesture_confidence))
                smoothed_label, smoothed_confidence = smooth_prediction(recent_preds)

                face_lms = (
                    face_results.multi_face_landmarks[0]
                    if getattr(face_results, "multi_face_landmarks", None)
                    else None
                )
                distress_score = calc_distress(face_lms, width, height)
                distress_flag = distress_score > DISTRESS_THRESHOLD
                append_distress_history(distress_history, distress_score, distress_flag)

                now = time.time()
                if smoothed_label != "NONE" and smoothed_confidence >= HAND_CONF_THRESHOLD:
                    payload = make_alert_payload(
                        smoothed_label,
                        smoothed_confidence,
                        distress_score,
                        distress_flag,
                    )
                    signature = (
                        f"{payload['MainGesture']}:{payload['Severity']}:"
                        f"{payload['DistressFlag']}"
                    )
                    if signature != last_alert_signature or now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                        log_alert(payload)
                        rprint(f"[bold red]ALERT:[/bold red] {json.dumps(payload, indent=2)}")
                        last_alert_signature = signature
                        last_alert_time = now
                        if requests is not None:
                            try:
                                requests.post(ALERT_ENDPOINT, json=payload, timeout=0.6)
                            except Exception as exc:  # pragma: no cover - network/runtime path
                                log_error(f"Alert POST failed: {exc}")

                overlay = draw_overlay(
                    frame,
                    hand_results,
                    face_lms,
                    smoothed_label,
                    smoothed_confidence,
                    distress_score,
                    distress_flag,
                )
                cv2.imshow("VERS Real-Time Demo (macOS)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rprint("[yellow]Exiting real-time loop.[/yellow]")
                    break

            except Exception as exc:
                log_error(f"Frame error: {exc}\n{traceback.format_exc()}")
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()

    rprint("[bold magenta]VERS demo finished.[/bold magenta]")


if __name__ == "__main__":
    main()
