"""Streamlit dashboard: the primary VERS demo surface for Phase-1."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path(__file__).resolve().parent.parent / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import cv2
import mediapipe as mp
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
from src.utils.alert_utils import make_alert_payload
from src.utils.data_utils import ensure_project_dirs, extract_hand_vector


def _trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
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
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="vers-streamlit-runtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
        with self._lock:
            self._thread = None
            self._status["running"] = False
            self._status["camera_active"] = False
            import numpy as np
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

    def _append_alert(self, payload: dict[str, Any]) -> None:
        self._alerts.appendleft(payload)
        self._status["last_alert"] = payload

    def _run_loop(self) -> None:
        ensure_project_dirs()
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
            with self._lock:
                self._status["error"] = (
                    "Unable to access webcam. Check camera permissions "
                    "(System Settings -> Privacy & Security -> Camera) and close other camera apps."
                )
                self._status["running"] = False
                self._status["camera_active"] = False
            return

        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        recent_preds: deque[tuple[str, float]] = deque(maxlen=SMOOTHING_WINDOW)
        last_alert_signature = ""
        last_alert_time = 0.0
        last_tick = time.perf_counter()

        with self._lock:
            self._status["running"] = True
            self._status["camera_active"] = True
            self._status["error"] = None

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue

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

                time.sleep(0.02)
        finally:
            cap.release()
            hands.close()
            face_mesh.close()
            with self._lock:
                self._status["running"] = False
                self._status["camera_active"] = False


@st.cache_resource
def get_runtime() -> DashboardRuntime:
    return DashboardRuntime()


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="VERS Dashboard", layout="wide")
    st.title("Vision-Based Emergency Response System (VERS)")
    st.caption("Primary Phase-1 demo dashboard for gesture recognition, distress scoring, and structured alerts.")

    runtime = get_runtime()

    st.session_state.setdefault("run_camera", False)

    model_error = None
    labels: list[str] = []
    try:
        labels = runtime.ensure_model_loaded()
    except Exception as exc:
        model_error = str(exc)

    st.sidebar.header("Configuration")
    conf_threshold = st.sidebar.slider("Gesture confidence threshold", 0.0, 1.0, 0.70, 0.05)
    distress_threshold = st.sidebar.slider("Distress threshold", 0.0, 0.2, float(DISTRESS_THRESHOLD), 0.005)
    st.sidebar.info(
        "Allow camera access for Terminal or your IDE in System Settings -> Privacy & Security -> Camera."
    )
    if labels:
        st.sidebar.markdown(f"**Recognized gestures**: {', '.join(labels)}")
    st.sidebar.caption("Legacy fallback remains available through `python web_vers.py`.")

    start_col, stop_col = st.sidebar.columns(2)
    start_clicked = start_col.button("Start Stream", use_container_width=True, disabled=model_error is not None)
    stop_clicked = stop_col.button("Stop Stream", use_container_width=True)

    st.sidebar.markdown("---")
    clear_clicked = st.sidebar.button("Clear Data", use_container_width=True)
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
    if snapshot.get("error"):
        st.error(snapshot["error"])

    if full_screen:
        st.subheader("Live Feed (Full Screen)")
        if snapshot.get("frame") is not None:
            st.image(snapshot["frame"], channels="RGB", use_column_width=True)
        else:
            st.info("Press 'Start Stream' to begin the live camera demo.")
    else:
        feed_col, side_col = st.columns([2.1, 1])
        with feed_col:
            st.subheader("Live Feed")
            if snapshot.get("frame") is not None:
                st.image(snapshot["frame"], channels="RGB", use_column_width=True)
            else:
                st.info("Press 'Start Stream' to begin the live camera demo.")

        with side_col:
            st.subheader("Current Readings")
            
            gesture_val = snapshot.get("gesture", "No gesture")
            if gesture_val in ["ACCIDENT", "EMERGENCY", "SOS"]:
                g_color = "#ff4b4b"
            elif gesture_val == "SAFETY":
                g_color = "#09ab3b"
            else:
                g_color = "inherit"
            
            st.markdown(f"**Gesture:**<br><span style='color:{g_color}; font-size: 2.2rem; font-weight: 600;'>{gesture_val}</span>", unsafe_allow_html=True)
            
            st.metric("Confidence", f"{float(snapshot.get('confidence', 0.0)):.2f}")
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
                st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No alerts emitted yet.")

        if snapshot.get("last_alert"):
            st.markdown("**Latest Alert Payload**")
            st.json(snapshot["last_alert"])

    if st.session_state.run_camera and not snapshot.get("error"):
        time.sleep(0.03)
        _trigger_rerun()


if __name__ == "__main__":
    main()
