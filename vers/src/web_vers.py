"""Legacy Flask dashboard for VERS.

Streamlit is the primary demo UI. This module keeps the older Flask dashboard
available as a fallback during demos and transition work.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
from flask import Flask, Response, jsonify, render_template
import mediapipe as mp

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
from src.utils.data_utils import ensure_project_dirs, extract_hand_vector, open_camera_capture

TEMPLATE_DIR = PROJECT_ROOT / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


def _legacy_alert_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": payload["Timestamp"],
        "gesture": payload["MainGesture"],
        "severity": payload["Severity"],
        "message": payload["Message"],
        "gesture_conf": payload["GestureConfidence"],
        "distress_score": payload["DistressScore"],
        "distress_flag": payload["DistressFlag"],
    }


class FlaskRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_bytes: bytes | None = None
        self._alerts: deque[dict[str, Any]] = deque(maxlen=20)
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
            "model_type": "sklearn_pipeline",
            "last_alert": None,
        }

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._status,
                "alerts": [_legacy_alert_view(payload) for payload in self._alerts],
                "last_alert": _legacy_alert_view(self._status["last_alert"]) if self._status["last_alert"] else None,
            }

    def frame_generator(self):
        while True:
            with self._lock:
                frame = self._frame_bytes
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)

    def _run_loop(self) -> None:
        ensure_project_dirs()
        try:
            model_bundle, labels = load_model()
        except Exception as exc:
            with self._lock:
                self._status["error"] = str(exc)
            return

        cap, _backend_info = open_camera_capture(max_index=4, warmup_reads=18)
        if cap is None:
            cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            with self._lock:
                self._status["error"] = "Unable to access webcam."
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
            self._status.update({"running": True, "camera_active": True, "labels": labels, "error": None})

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
                if hand_vector is not None:
                    raw_label, raw_conf = predict_gesture(model_bundle, hand_vector)
                recent_preds.append((raw_label, raw_conf))
                smooth_label, smooth_conf = smooth_prediction(recent_preds)

                face_results = face_mesh.process(rgb)
                face_lms = (
                    face_results.multi_face_landmarks[0]
                    if getattr(face_results, "multi_face_landmarks", None)
                    else None
                )
                distress_score = calc_distress(face_lms, width, height)
                distress_flag = distress_score > DISTRESS_THRESHOLD

                payload = None
                now = time.time()
                if smooth_label != "NONE" and smooth_conf >= HAND_CONF_THRESHOLD:
                    payload = make_alert_payload(smooth_label, smooth_conf, distress_score, distress_flag)
                    signature = (
                        f"{payload['MainGesture']}:{payload['Severity']}:"
                        f"{payload['DistressFlag']}"
                    )
                    if signature != last_alert_signature or now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                        with self._lock:
                            self._alerts.appendleft(payload)
                            self._status["last_alert"] = payload
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

                ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                tick = time.perf_counter()
                fps = 1.0 / max(tick - last_tick, 1e-6)
                last_tick = tick

                with self._lock:
                    self._frame_bytes = encoded.tobytes() if ok else None
                    self._status.update(
                        {
                            "running": True,
                            "camera_active": True,
                            "gesture": smooth_label if smooth_label != "NONE" else "No gesture",
                            "confidence": smooth_conf if smooth_label != "NONE" else 0.0,
                            "distress_score": distress_score,
                            "distress_flag": distress_flag,
                            "fps": fps,
                            "labels": labels,
                            "last_alert": payload or self._status["last_alert"],
                            "error": None,
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


runtime = FlaskRuntime()


@app.route("/")
def index():
    runtime.start()
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    runtime.start()
    return Response(runtime.frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    runtime.start()
    return jsonify(runtime.status())


def main() -> None:
    runtime.start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
