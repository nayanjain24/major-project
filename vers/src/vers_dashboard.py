"""Streamlit dashboard: the primary VERS demo surface for Phase-1."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime_vers import (
    ALERT_COOLDOWN_SECONDS,
    DISTRESS_THRESHOLD,
    HAND_CONF_THRESHOLD,
    SMOOTHING_WINDOW,
    calc_distress,
    draw_overlay,
    load_model,
    predict_gesture,
    smooth_prediction,
)
from src.utils.alert_utils import ALERT_MAP, log_error, make_alert_payload
from src.utils.data_utils import ensure_project_dirs, extract_hand_vector, open_camera_capture


def _trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


class DashboardRuntime:
    """Background webcam worker so Streamlit can offer responsive start/stop controls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None
        self._alerts: deque[dict[str, Any]] = deque(maxlen=15)
        self._status: dict[str, Any] = {
            "running": False,
            "camera_active": False,
            "gesture": "No gesture",
            "confidence": 0.0,
            "distress_score": 0.0,
            "distress_flag": False,
            "fps": 0.0,
            "error": None,
            "labels": [],
            "last_alert": None,
            "starting": False,
        }
        self._conf_threshold = HAND_CONF_THRESHOLD
        self._distress_threshold = DISTRESS_THRESHOLD
        self._model_bundle: dict[str, Any] | None = None
        self._labels: list[str] = []

    def ensure_model_loaded(self) -> list[str]:
        with self._lock:
            if self._model_bundle is None:
                self._model_bundle, self._labels = load_model()
                self._status["labels"] = list(self._labels)
            return list(self._labels)

    def model_bundle(self) -> dict[str, Any]:
        self.ensure_model_loaded()
        with self._lock:
            if self._model_bundle is None:
                raise RuntimeError("Model bundle is not available.")
            return self._model_bundle

    def configure(self, conf_threshold: float, distress_threshold: float) -> None:
        with self._lock:
            self._conf_threshold = conf_threshold
            self._distress_threshold = distress_threshold

    def start(self, conf_threshold: float, distress_threshold: float) -> None:
        self.ensure_model_loaded()
        self.configure(conf_threshold, distress_threshold)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._status["error"] = None
            self._status["starting"] = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="vers-streamlit-runtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Force release the camera immediately to break any blocking cap.read()
        with self._lock:
            if self._status.get("camera_active"):
                # We try a 'soft' release by signaling the thread, but if it's lagging,
                # the thread will catch the _stop_event.
                pass 
        
        thread = self._thread
        if thread is not None and thread.is_alive():
            # If we were to call cap.release() here, we might cause a race in the thread.
            # Instead, we rely on the _stop_event being checked before the next cap.read().
            # To make it 'buttery smooth', we'll ensure the thread doesn't sleep if it's stopping.
            thread.join(timeout=0.5)
        with self._lock:
            self._thread = None
            self._status["running"] = False
            self._status["camera_active"] = False
            self._status["starting"] = False
            self._status["gesture"] = "No gesture"
            self._status["confidence"] = 0.0
            self._status["distress_score"] = 0.0
            self._status["distress_flag"] = False
            self._status["fps"] = 0.0
            self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return {
                **self._status,
                "frame": frame,
                "alerts": list(self._alerts),
            }

    def clear_data(self) -> None:
        with self._lock:
            self._alerts.clear()
            self._status["last_alert"] = None
            self._status["gesture"] = "No gesture"
            self._status["confidence"] = 0.0
            self._status["distress_score"] = 0.0
            self._status["distress_flag"] = False
            self._status["fps"] = 0.0

    def _append_alert(self, payload: dict[str, Any]) -> None:
        self._alerts.appendleft(payload)
        self._status["last_alert"] = payload

    def add_alert(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if not self._alerts or self._alerts[0] != payload:
                self._append_alert(payload)

    def _run_loop(self) -> None:
        ensure_project_dirs()
        cap, backend_info = open_camera_capture(max_index=4, warmup_reads=18)
        if cap is None:
            cap = cv2.VideoCapture(0)
            backend_info = "DEFAULT:0"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            with self._lock:
                self._status["error"] = (
                    "Unable to access webcam. Check camera permissions "
                    "(System Settings -> Privacy & Security -> Camera) and close other camera apps."
                )
                self._status["running"] = False
                self._status["camera_active"] = False
                self._status["starting"] = False
            return

        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,  # Optimized for performance (0 is fastest)
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Simplified mesh for higher FPS
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        recent_preds: deque[tuple[str, float]] = deque(maxlen=SMOOTHING_WINDOW)
        last_alert_signature = ""
        last_alert_time = 0.0
        last_tick = time.perf_counter()
        consecutive_capture_failures = 0
        max_capture_failures = 45

        with self._lock:
            self._status["running"] = True
            self._status["camera_active"] = True
            self._status["starting"] = False
            self._status["error"] = None

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_capture_failures += 1
                    if consecutive_capture_failures >= max_capture_failures:
                        raise RuntimeError(
                            "Camera opened but no frames are being delivered. "
                            "Close other camera apps, then verify Camera permission for "
                            "both Terminal and Codex/IDE, and restart stream."
                        )
                    time.sleep(0.03)
                    continue
                consecutive_capture_failures = 0

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape

                hand_results = hands.process(rgb)
                hand_vector = extract_hand_vector(hand_results)
                raw_label, raw_conf = "NONE", 0.0
                if hand_vector is not None and self._model_bundle is not None:
                    raw_label, raw_conf = predict_gesture(self._model_bundle, hand_vector)
                recent_preds.append((raw_label, raw_conf))
                smooth_label, smooth_conf = smooth_prediction(recent_preds)

                face_results = face_mesh.process(rgb)
                face_lms = (
                    face_results.multi_face_landmarks[0]
                    if getattr(face_results, "multi_face_landmarks", None)
                    else None
                )
                distress_score = calc_distress(face_lms, width, height)

                with self._lock:
                    distress_threshold = self._distress_threshold
                    conf_threshold = self._conf_threshold

                distress_flag = distress_score > distress_threshold
                payload = None
                now = time.time()
                if smooth_label != "NONE" and smooth_conf >= conf_threshold:
                    payload = make_alert_payload(smooth_label, smooth_conf, distress_score, distress_flag)
                    signature = (
                        f"{payload['MainGesture']}:{payload['Severity']}:"
                        f"{payload['DistressFlag']}"
                    )
                    if signature != last_alert_signature or now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                        with self._lock:
                            self._append_alert(payload)
                        last_alert_signature = signature
                        last_alert_time = now

                overlay = draw_overlay(
                    frame,
                    hand_results,
                    face_lms,
                    smooth_label,
                    smooth_conf,
                    distress_score,
                    distress_flag,
                    conf_threshold=conf_threshold,
                )
                frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                tick = time.perf_counter()
                fps = 1.0 / max(tick - last_tick, 1e-6)
                last_tick = tick

                with self._lock:
                    self._frame = frame_rgb
                    self._status.update(
                        {
                            "running": True,
                            "camera_active": True,
                            "gesture": smooth_label if smooth_label != "NONE" else "No gesture",
                            "confidence": smooth_conf if smooth_label != "NONE" else 0.0,
                            "distress_score": distress_score,
                            "distress_flag": distress_flag,
                            "fps": fps,
                            "error": None,
                            "labels": list(self._labels),
                            "last_alert": payload or self._status.get("last_alert"),
                        }
                    )

                # Decreased sleep for 'buttery smooth' 30+ FPS capability
                # We only sleep long enough to prevent 100% CPU pinning
                time.sleep(0.005)
        except Exception as exc:
            exc_text = str(exc).strip() or type(exc).__name__
            log_error(f"Dashboard runtime loop failure [{backend_info}]: {exc_text}")
            with self._lock:
                self._status["error"] = f"Camera worker crashed: {exc_text}"
                self._status["running"] = False
                self._status["camera_active"] = False
        finally:
            cap.release()
            hands.close()
            face_mesh.close()
            with self._lock:
                self._status["running"] = False
                self._status["camera_active"] = False
                self._status["starting"] = False
                self._status["gesture"] = "No gesture"
                self._status["confidence"] = 0.0
                self._status["distress_score"] = 0.0
                self._status["distress_flag"] = False
                self._status["fps"] = 0.0
                self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)


@st.cache_resource
def get_runtime() -> DashboardRuntime:
    return DashboardRuntime()


def _severity_color(label: str) -> str:
    severity = ALERT_MAP.get(str(label).upper(), ALERT_MAP["NONE"]).get("severity", "Low")
    if severity in ("Critical", "High"):
        return "#ff4b4b"
    if severity == "Medium":
        return "#ff9f43"
    if label == "No gesture":
        return "inherit"
    return "#09ab3b"


def _analyze_browser_capture(
    image_bytes: bytes,
    model_bundle: dict[str, Any],
    conf_threshold: float,
    distress_threshold: float,
) -> dict[str, Any]:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode captured image.")

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    with mp.solutions.hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    ) as hands, mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

    hand_vector = extract_hand_vector(hand_results)
    label, confidence = ("NONE", 0.0)
    if hand_vector is not None:
        label, confidence = predict_gesture(model_bundle, hand_vector)

    face_lms = (
        face_results.multi_face_landmarks[0]
        if getattr(face_results, "multi_face_landmarks", None)
        else None
    )
    distress_score = calc_distress(face_lms, width, height)
    distress_flag = distress_score > distress_threshold
    accepted = (label != "NONE" and confidence >= conf_threshold)

    payload = (
        make_alert_payload(label, confidence, distress_score, distress_flag)
        if accepted
        else None
    )
    overlay = draw_overlay(
        frame,
        hand_results,
        face_lms,
        label,
        confidence,
        distress_score,
        distress_flag,
        conf_threshold=conf_threshold,
    )
    return {
        "frame": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        "gesture": label if accepted else "No gesture",
        "confidence": confidence if accepted else 0.0,
        "distress_score": distress_score,
        "distress_flag": distress_flag,
        "payload": payload,
    }


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="VERS Dashboard", layout="wide")
    st.title("Vision-Based Emergency Response System (VERS)")
    st.caption("Primary Phase-1 demo dashboard for gesture recognition, distress scoring, and structured alerts.")

    runtime = get_runtime()

    st.session_state.setdefault("run_camera", False)
    st.session_state.setdefault("fallback_result", None)

    model_error = None
    labels: list[str] = []
    try:
        labels = runtime.ensure_model_loaded()
    except Exception as exc:
        model_error = str(exc).strip() or type(exc).__name__
        st.session_state.run_camera = False

    st.sidebar.header("Configuration")
    conf_threshold = 0.0
    distress_threshold = st.sidebar.slider("Distress threshold", 0.0, 0.2, float(DISTRESS_THRESHOLD), 0.005)
    st.sidebar.info(
        "Allow camera access for Terminal or your IDE in System Settings -> Privacy & Security -> Camera."
    )
    
    st.sidebar.markdown("### Gesture Guide (Custom)")
    st.sidebar.markdown("""
    - ✋ **SOS**: Full Open Hand (5 fingers).
    - ✌️ **EMERGENCY**: 2 Fingers (V-shape).
    - ✊ **ACCIDENT**: Solid Fist (all fingers closed).
    - 🖐️ **MEDICAL**: 4 Fingers (Index to Pinky out, Thumb folded).
    - 👍 **SAFE**: Thumbs Up.
    """)
    
    st.sidebar.markdown("### 🛠️ Calibration (Recommended)")
    st.sidebar.markdown("""
    Synthetic data is a start, but for **perfect accuracy**, you should calibrate the model using your own hand.
    
    **To calibrate:**
    1. Stop the current Streamlit app (`Ctrl+C` in Terminal).
    2. Run this command:
    ```bash
    python src/orchestrate.py --calibrate
    ```
    3. Follow the sequence to record HELP, MEDICAL, DANGER, and SOS.
    """)

    if labels:
        st.sidebar.markdown(f"**Recognized gestures**: {', '.join(labels)}")
    st.sidebar.caption("Legacy fallback remains available through `python web_vers.py`.")

    pre_snapshot = runtime.snapshot()
    is_running = bool(pre_snapshot.get("running"))

    start_col, stop_col = st.sidebar.columns(2)
    start_clicked = start_col.button(
        "Start Stream",
        disabled=(model_error is not None or is_running),
    )
    stop_clicked = stop_col.button(
        "Stop Stream",
        disabled=not is_running,
    )

    st.sidebar.markdown("---")
    clear_clicked = st.sidebar.button("Clear Data")
    if clear_clicked:
        runtime.clear_data()

    full_screen = st.sidebar.checkbox("Full Screen Video Mode", value=False)

    if start_clicked:
        runtime.start(conf_threshold, distress_threshold)
        st.session_state.run_camera = True

    if stop_clicked:
        runtime.stop()
        st.session_state.run_camera = False

    if st.session_state.run_camera:
        runtime.configure(conf_threshold, distress_threshold)

    if model_error is not None:
        st.error(model_error)

    snapshot = runtime.snapshot()
    is_starting = bool(snapshot.get("starting"))
    
    if st.session_state.run_camera and not snapshot.get("running") and not is_starting:
        st.session_state.run_camera = False
    if snapshot.get("error"):
        st.error(snapshot["error"])

    if full_screen:
        st.subheader("Live Feed (Full Screen)")
        if snapshot.get("frame") is not None:
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(snapshot["frame"], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            st.image(buffer.tobytes(), use_column_width=True)
        else:
            st.info("Press 'Start Stream' to begin the live camera demo.")
    else:
        feed_col, side_col = st.columns([2.1, 1])
        with feed_col:
            st.subheader("Live Feed")
            if snapshot.get("frame") is not None:
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(snapshot["frame"], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                st.image(buffer.tobytes(), use_column_width=True)
            else:
                st.info("Press 'Start Stream' to begin the live camera demo.")

        with side_col:
            st.subheader("Current Readings")

            gesture_val = snapshot.get("gesture", "No gesture")
            g_color = _severity_color(gesture_val)

            st.markdown(
                f"**Gesture:**<br><span style='color:{g_color}; font-size: 2.2rem; font-weight: 600;'>{gesture_val}</span>",
                unsafe_allow_html=True,
            )

            # st.metric("Confidence", f"{float(snapshot.get('confidence', 0.0)):.2f}")
            distress_value = float(snapshot.get("distress_score", 0.0))
            distress_delta = "High" if snapshot.get("distress_flag") else "Normal"
            st.metric("Distress Score", f"{distress_value:.3f}", delta=distress_delta)
            st.metric("FPS", f"{float(snapshot.get('fps', 0.0)):.1f}")

            st.markdown("---")
            st.subheader("Recent Alerts")
            alerts = snapshot.get("alerts", [])
            if alerts:
                alert_rows = [
                    {
                        "Timestamp": payload["Timestamp"],
                        "Gesture": payload["MainGesture"],
                        "Severity": payload["Severity"],
                        "Confidence": payload["GestureConfidence"],
                        "DistressScore": payload["DistressScore"],
                        "DistressFlag": payload["DistressFlag"],
                        "Message": payload["Message"],
                    }
                    for payload in alerts
                ]
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
            else:
                st.info("No alerts emitted yet.")

        if snapshot.get("last_alert"):
            st.markdown("**Latest Alert Payload**")
            st.json(snapshot["last_alert"])

    if snapshot.get("error"):
        st.markdown("---")
        st.subheader("Browser Camera Fallback")
        st.caption(
            "If backend OpenCV camera access is blocked, capture a browser frame and run one-shot inference."
        )
        fallback_capture = st.camera_input(
            "Capture Frame (Browser Permission Mode)",
            help="This uses the browser camera permission flow instead of backend OpenCV access.",
        )
        analyze_clicked = st.button(
            "Analyze Captured Frame",
            disabled=fallback_capture is None or model_error is not None,
        )

        if analyze_clicked and fallback_capture is not None:
            try:
                fallback_result = _analyze_browser_capture(
                    fallback_capture.getvalue(),
                    runtime.model_bundle(),
                    conf_threshold,
                    distress_threshold,
                )
                st.session_state.fallback_result = fallback_result
                payload = fallback_result.get("payload")
                if payload is not None:
                    runtime.add_alert(payload)
            except Exception as exc:
                log_error(f"Browser fallback inference failed: {exc}")
                st.session_state.fallback_result = None
                st.error(f"Fallback analysis failed: {exc}")

        fallback_result = st.session_state.get("fallback_result")
        if fallback_result:
            st.image(fallback_result["frame"], channels="RGB", caption="Fallback Analysis Preview")
            metric_col1, metric_col3 = st.columns(2)
            metric_col1.metric("Gesture", fallback_result["gesture"])
            metric_col3.metric("Distress Score", f"{float(fallback_result['distress_score']):.3f}")
            if fallback_result.get("payload"):
                st.success("Alert generated from fallback capture.")
                st.json(fallback_result["payload"])
            else:
                st.info("No alert emitted from this capture (gesture confidence below threshold).")

    if st.session_state.run_camera and not snapshot.get("error"):
        # Faster rerun for higher visual frame rate (avoid artificial lag)
        time.sleep(0.01)
        _trigger_rerun()


if __name__ == "__main__":
    main()
