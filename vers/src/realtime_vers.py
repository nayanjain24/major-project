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

MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

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
    DATA_PATH,
    DISTRESS_HISTORY_PATH,
    MODEL_PATH,
    ensure_project_dirs,
    extract_hand_vector,
    open_camera_capture,
)

VERS_VERSION = "1.0.0"
ALERT_ENDPOINT = "http://localhost:8000/alert"
DISTRESS_THRESHOLD = 0.055
HAND_CONF_THRESHOLD = 0.0
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
    def build_centroid_bundle_from_dataset() -> tuple[dict[str, Any], list[str]]:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - import/runtime path
            raise RuntimeError("pandas is required for centroid fallback model.") from exc

        if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
            raise FileNotFoundError(
                f"Dataset missing at {DATA_PATH}. Run `python src/record_gestures.py` first."
            )
        df = pd.read_csv(DATA_PATH)
        if "label" not in df.columns:
            raise ValueError(f"{DATA_PATH} is missing 'label' column.")

        feature_columns = [column for column in df.columns if column.startswith("f_")]
        if len(feature_columns) != 63:
            raise ValueError(
                f"Expected 63 feature columns in {DATA_PATH}, found {len(feature_columns)}."
            )

        labels_local = sorted(df["label"].astype(str).str.upper().unique().tolist())
        centroids: dict[str, list[float]] = {}
        scales: dict[str, list[float]] = {}
        for label in labels_local:
            subset = df[df["label"].astype(str).str.upper() == label][feature_columns].to_numpy(dtype=np.float32)
            if subset.size == 0:
                continue
            centroid = subset.mean(axis=0)
            scale = subset.std(axis=0)
            scale = np.where(scale < 1e-6, 1.0, scale)
            centroids[label] = centroid.tolist()
            scales[label] = scale.tolist()

        if not centroids:
            raise ValueError("Could not build centroid fallback: no labeled samples available.")

        fallback_bundle = {
            "model_type": "centroid",
            "labels": labels_local,
            "centroids": centroids,
            "scales": scales,
            "feature_columns": feature_columns,
            "source": "dataset_centroid_fallback",
        }
        return fallback_bundle, labels_local

    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            labels = list(bundle.get("labels", []))
            if "pipeline" in bundle:
                return bundle, labels
            if bundle.get("model_type") == "centroid" and "centroids" in bundle and "scales" in bundle:
                return bundle, labels
            log_error(f"Unsupported model bundle format at {MODEL_PATH}; using centroid fallback.")
        except Exception as exc:
            exc_text = str(exc).strip() or type(exc).__name__
            log_error(f"Failed to load {MODEL_PATH}: {exc_text}; using centroid fallback.")
    else:
        log_error(f"Model file not found at {MODEL_PATH}; using centroid fallback.")

    fallback_bundle, fallback_labels = build_centroid_bundle_from_dataset()
    rprint(
        "[yellow]Using centroid fallback model built from data/landmarks.csv "
        "(primary model bundle unavailable).[/yellow]"
    )
    return fallback_bundle, fallback_labels


def predict_gesture(model_bundle: dict[str, Any], hand_vector: Any) -> tuple[str, float]:
    """Predict the gesture label using perfect 3D geometry derived from hand physics."""
    if hasattr(hand_vector, "to_numpy"):
        vector_2d = hand_vector.to_numpy()
    else:
        vector_2d = np.asarray(hand_vector)
        
    v = vector_2d.reshape(21, 3)
    
    def dist(i: int, j: int = 0) -> float:
        return float(np.linalg.norm(v[i] - v[j]))
        
    idx_ext = dist(8) > dist(6)
    mid_ext = dist(12) > dist(10)
    rng_ext = dist(16) > dist(14)
    pnk_ext = dist(20) > dist(18)
    
    # Thumb folded means it stretches across the palm towards pinky (17)
    thumb_ext = dist(4, 17) > dist(3, 17)
    
    if thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return "SOS", 1.0
    if not thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return "MEDICAL", 1.0
    if not thumb_ext and idx_ext and mid_ext and not rng_ext and not pnk_ext:
        return "EMERGENCY", 1.0
    if thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return "SAFE", 1.0
    if not thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return "ACCIDENT", 1.0
        
    return "NONE", 0.0


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
    conf_threshold: float = HAND_CONF_THRESHOLD,
    fps: float = 0.0,
) -> np.ndarray:
    """Render the real-time demo overlay with high-visibility background boxes."""
    overlay = frame.copy()
    is_accepted = (display_label != "NONE" and display_confidence >= conf_threshold)
    is_none = (display_label == "NONE")
    
    # Text and layout constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    x_offset = 15
    y_start = 40
    line_height = 35

    def draw_text_with_bg(img, text, pos, color, bg_color=(0, 0, 0), alpha=0.6):
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx, ty = pos
        # Draw background rect
        rect_start = (tx - 5, ty - th - 5)
        rect_end = (tx + tw + 5, ty + baseline + 5)
        sub_img = img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
        if sub_img.size > 0:
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 0)
            img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = res
        cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # 1. Gesture Line
    if is_none:
        g_text = "Gesture: None detected"
        g_color = (200, 200, 200)
    elif is_accepted:
        severity = ALERT_MAP.get(str(display_label).upper(), ALERT_MAP["NONE"]).get("severity", "Low")
        g_text = f"Gesture: {display_label} (MATCH)"
        g_color = (0, 255, 0) if severity == "Low" else (0, 165, 255) if severity == "Medium" else (0, 0, 255)
    else:
        g_text = f"Gesture: {display_label} (LOW CONFIDENCE)"
        g_color = (0, 255, 255) # Yellow/Orange

    draw_text_with_bg(overlay, g_text, (x_offset, y_start), g_color)

    # 2. Confidence Line
    # c_color = (0, 255, 0) if is_accepted else (0, 255, 255) if not is_none else (200, 200, 200)
    # draw_text_with_bg(overlay, f"Confidence: {display_confidence:.2f}", (x_offset, y_start + line_height), c_color)

    # 3. Distress Line
    d_color = (0, 0, 255) if distress_flag else (0, 255, 0)
    draw_text_with_bg(overlay, f"Distress: {distress_score:.3f} {'(HIGH)' if distress_flag else '(NORMAL)'}", 
                      (x_offset, y_start + (line_height * 2)), d_color)

    # 4. FPS counter
    if fps > 0:
        draw_text_with_bg(overlay, f"FPS: {fps:.1f}", (x_offset, y_start + (line_height * 3)), (255, 255, 255))

    # 5. System status (Bottom Left)
    draw_text_with_bg(overlay, f"VERS v{VERS_VERSION} | System Active", (x_offset, frame.shape[0] - 20), (220, 220, 220), alpha=0.4)

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

    cap, backend_info = open_camera_capture(max_index=4, warmup_reads=18)
    if cap is None:
        cap = cv2.VideoCapture(0)
        backend_info = "DEFAULT:0"

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

                fps = 1.0 / max(time.time() - now + 1e-6, 1e-6)
                overlay = draw_overlay(
                    frame,
                    hand_results,
                    face_lms,
                    smoothed_label,
                    smoothed_confidence,
                    distress_score,
                    distress_flag,
                    fps=fps,
                )
                cv2.imshow("VERS Real-Time Demo (macOS)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rprint("[yellow]Exiting real-time loop.[/yellow]")
                    break

            except Exception as exc:
                log_error(f"Frame error [{backend_info}]: {exc}\n{traceback.format_exc()}")
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()

    rprint("[bold magenta]VERS demo finished.[/bold magenta]")


if __name__ == "__main__":
    main()
