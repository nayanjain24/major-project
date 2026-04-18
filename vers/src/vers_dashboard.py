"""Streamlit dashboard: VERS v2.0 Multimodal AI Command Center.

Integrates:
  - Vision: MediaPipe hand tracking + DeepFace facial emotion recognition
  - Intelligence: Temporal sequence smoothing + Multimodal severity fusion
  - Services: Async TTS voice alerts + Simulated GPS alert dispatch
"""

from __future__ import annotations

import base64
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

# --- Legacy imports (still needed for model loading and overlay drawing) ---
from src.realtime_vers import (
    ALERT_COOLDOWN_SECONDS,
    DISTRESS_THRESHOLD,
    HAND_CONF_THRESHOLD,
    SMOOTHING_WINDOW,
    calc_distress,
    draw_overlay,
    load_model,
)
from src.utils.alert_utils import ALERT_MAP, log_error, make_alert_payload
from src.utils.data_utils import ensure_project_dirs, extract_hand_vector, open_camera_capture

# --- v2.0 Modular Architecture imports ---
from src.vision.gesture_tracker import predict_gesture
from src.vision.emotion_model import analyze_emotion
from src.intelligence.temporal_smoothing import TemporalSmoother
from src.intelligence.multimodal_fusion import fuse, FusionResult
from src.services.alert_dispatcher import dispatch as dispatch_alert
from src.services import voice_tts


def _trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


class DashboardRuntime:
    """Background webcam worker with v2.0 multimodal AI pipeline."""

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
            "dominant_emotion": "neutral",
            "emotion_distress": 0.0,
            "severity_score": 0.0,
            "threat_level": "NONE",
            "location": None,
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
        # Start TTS daemon once
        voice_tts.start()

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
        """Core v2.0 multimodal AI pipeline — runs in a background thread."""
        ensure_project_dirs()
        cap, backend_info = open_camera_capture(max_index=4, warmup_reads=18)
        if cap is None:
            cap = cv2.VideoCapture(0)
            backend_info = "DEFAULT:0"

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
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
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # v2.0: Temporal smoother instead of raw deque
        smoother = TemporalSmoother(window_size=7, min_votes=3)
        last_tick = time.perf_counter()
        consecutive_capture_failures = 0
        max_capture_failures = 45
        frame_counter = 0
        # Cached emotion result (updated every 3rd frame for FPS)
        cached_emotion: dict[str, Any] = {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "distress_contribution": 0.0,
        }

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
                frame_counter += 1

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape

                # --- HAND GESTURE (every frame) ---
                hand_results = hands.process(rgb)
                hand_vector = extract_hand_vector(hand_results)
                raw_label, raw_conf = "NONE", 0.0
                if hand_vector is not None:
                    raw_label, raw_conf = predict_gesture(hand_vector)
                smoother.push(raw_label, raw_conf)
                smooth_label, smooth_conf = smoother.smoothed()

                # --- FACE MESH + HEURISTIC DISTRESS (every frame) ---
                face_results = face_mesh.process(rgb)
                face_lms = (
                    face_results.multi_face_landmarks[0]
                    if getattr(face_results, "multi_face_landmarks", None)
                    else None
                )
                distress_score = calc_distress(face_lms, width, height)

                # --- DEEPFACE EMOTION (every 3rd frame for performance) ---
                if frame_counter % 3 == 0:
                    try:
                        cached_emotion = analyze_emotion(rgb)
                    except Exception:
                        pass  # Keep cached value

                with self._lock:
                    distress_threshold = self._distress_threshold

                distress_flag = distress_score > distress_threshold

                # --- MULTIMODAL FUSION ---
                fusion: FusionResult = fuse(
                    gesture_label=smooth_label,
                    gesture_confidence=smooth_conf,
                    dominant_emotion=cached_emotion.get("dominant_emotion", "neutral"),
                    emotion_distress=cached_emotion.get("distress_contribution", 0.0),
                )

                # --- ALERT DISPATCH (with TTS + location) ---
                payload = dispatch_alert(
                    gesture_label=fusion.gesture_label,
                    gesture_confidence=fusion.gesture_confidence,
                    dominant_emotion=fusion.dominant_emotion,
                    emotion_distress=fusion.emotion_distress,
                    severity_score=fusion.severity_score,
                    threat_level=fusion.threat_level,
                    distress_flag=distress_flag,
                    enable_tts=True,
                )
                if payload is not None:
                    with self._lock:
                        self._append_alert(payload)

                # --- OVERLAY DRAWING ---
                overlay = draw_overlay(
                    frame,
                    hand_results,
                    face_lms,
                    smooth_label,
                    smooth_conf,
                    distress_score,
                    distress_flag,
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
                            "dominant_emotion": fusion.dominant_emotion,
                            "emotion_distress": fusion.emotion_distress,
                            "severity_score": fusion.severity_score,
                            "threat_level": fusion.threat_level,
                            "location": payload.get("Location") if payload else self._status.get("location"),
                            "fps": fps,
                            "error": None,
                            "labels": list(self._labels),
                            "last_alert": payload or self._status.get("last_alert"),
                        }
                    )

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
                self._status["dominant_emotion"] = "neutral"
                self._status["emotion_distress"] = 0.0
                self._status["severity_score"] = 0.0
                self._status["threat_level"] = "NONE"
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
    st.set_page_config(page_title="VERS Command Center", layout="wide", page_icon="🚨", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Glassmorphism Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(30, 30, 42, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
            padding: 1.2rem;
            border-radius: 12px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.6);
            border-color: rgba(255,255,255,0.25);
        }

        /* Ambient Sidebar Gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e0e12 0%, #151520 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        header {
            background: transparent !important;
        }

        /* Dynamic Buttons */
        [data-testid="stButton"] button {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        [data-testid="stButton"] button:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.15);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("VERS v2.0 — Multimodal AI Command Center")
    st.caption("Real-time gesture + emotion fusion · Temporal smoothing · Voice alerts · Simulated GPS dispatch")

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

    # ---- Helper: render the video feed ----
    def _render_feed(frame_data):
        if frame_data is not None:
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_img = base64.b64encode(buffer).decode("utf-8")
            st.markdown(f'<img src="data:image/jpeg;base64,{b64_img}" style="width:100%; border-radius:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.4);">', unsafe_allow_html=True)
        else:
            st.info("Press 'Start Stream' to begin the live camera demo.")

    # ---- Helper: threat colour mapping ----
    def _threat_color(level: str) -> str:
        return {"CRITICAL": "#ff1744", "HIGH": "#ff5722", "MEDIUM": "#ff9800", "LOW": "#4caf50"}.get(level, "#90a4ae")

    if full_screen:
        st.subheader("Live Feed (Full Screen)")
        _render_feed(snapshot.get("frame"))
    else:
        feed_col, side_col = st.columns([2.1, 1])
        with feed_col:
            st.subheader("Live Feed")
            _render_feed(snapshot.get("frame"))

        with side_col:
            st.subheader("Current Readings")

            gesture_val = snapshot.get("gesture", "No gesture")
            g_color = _severity_color(gesture_val)
            st.markdown(
                f"**Gesture:**<br><span style='color:{g_color}; font-size: 2.2rem; font-weight: 600;'>{gesture_val}</span>",
                unsafe_allow_html=True,
            )

            # --- v2.0: Emotion + Severity + Threat Level ---
            emotion_val = snapshot.get("dominant_emotion", "neutral").capitalize()
            emotion_distress = float(snapshot.get("emotion_distress", 0.0))
            severity_score = float(snapshot.get("severity_score", 0.0))
            threat_level = snapshot.get("threat_level", "NONE")
            t_color = _threat_color(threat_level)

            st.markdown(
                f"**Emotion:** <span style='font-size:1.3rem;'>{emotion_val}</span> &nbsp; "
                f"<span style='color:#888; font-size:0.9rem;'>distress: {emotion_distress:.2f}</span>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Threat Level:** <span style='color:{t_color}; font-size:1.8rem; font-weight:700;'>{threat_level}</span>",
                unsafe_allow_html=True,
            )

            # Severity progress bar
            st.markdown(f"**Severity Score:** `{severity_score:.3f}`")
            st.progress(min(severity_score, 1.0))

            distress_value = float(snapshot.get("distress_score", 0.0))
            distress_delta = "⚠ High" if snapshot.get("distress_flag") else "✓ Normal"
            st.metric("Distress Score", f"{distress_value:.3f}", delta=distress_delta)
            st.metric("FPS", f"{float(snapshot.get('fps', 0.0)):.1f}")

    # --- v2.0: Tabs for Alerts and JSON Inspector ---
    st.markdown("---")
    tab_alerts, tab_json = st.tabs(["📋 Recent Alerts", "🔍 JSON Inspector"])

    with tab_alerts:
        alerts = snapshot.get("alerts", [])
        if alerts:
            alert_rows = [
                {
                    "Time": p.get("Timestamp", "")[:19],
                    "Gesture": p.get("MainGesture", ""),
                    "Emotion": p.get("DominantEmotion", "n/a"),
                    "Threat": p.get("ThreatLevel", ""),
                    "Severity": p.get("SeverityScore", 0.0),
                    "Message": p.get("Message", ""),
                }
                for p in alerts
            ]
            st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)
        else:
            st.info("No alerts emitted yet.")

    with tab_json:
        if snapshot.get("last_alert"):
            st.json(snapshot["last_alert"])
        else:
            st.info("No JSON payload available yet. Start the stream and perform a gesture.")

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
        time.sleep(0.02)
        _trigger_rerun()


if __name__ == "__main__":
    main()
