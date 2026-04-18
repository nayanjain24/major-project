"""Streamlit responder dashboard for the Vision-Based Emergency Response System (VERS).

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from app.vers_engine import (
    EmergencyGestureRecognizer,
    FaceDistressAnalyzer,
    VERSAlertEngine,
    calm_distress,
    enhance_hand_detection_frame,
    resize_for_inference,
)


ALERT_LOG_PATH = Path("runtime/vers_alerts.jsonl")
VIDEO_PRESETS: Dict[str, Dict[str, int]] = {
    "HD 720p": {"width": 1280, "height": 720, "frame_rate": 24},
    "Full HD 1080p": {"width": 1920, "height": 1080, "frame_rate": 20},
    "Balanced 540p": {"width": 960, "height": 540, "frame_rate": 24},
}


def default_latest_state() -> Dict[str, Any]:
    return {
        "gesture": "UNKNOWN",
        "gesture_confidence": 0.0,
        "distress_label": "CALM",
        "distress_score": 0.0,
        "incident_type": "Monitoring",
        "severity": "low",
        "status_banner": "LIVE MONITORING: MONITORING",
        "message": "System is waiting for a stable gesture or distress cue.",
        "recommended_action": "No action required yet.",
        "stable_frames": 0,
        "active_signal": "monitoring",
        "alerts": [],
        "alert_count": 0,
        "new_alert": None,
        "log_path": str(ALERT_LOG_PATH),
    }


def build_media_constraints(width: int, height: int, frame_rate: int) -> Dict[str, Any]:
    return {
        "video": {
            "width": {"ideal": width, "min": 640},
            "height": {"ideal": height, "min": 360},
            "frameRate": {"ideal": frame_rate, "min": 15},
        },
        "audio": False,
    }


def status_theme(incident_type: str) -> Dict[str, Any]:
    normalized = (incident_type or "").strip().lower()
    if normalized in {"safe / resolved", "monitoring"}:
        return {
            "overlay_bgr": (46, 204, 113),
            "bg": "#edf9f0",
            "border": "#2c9b5f",
            "fg": "#104127",
        }
    return {
        "overlay_bgr": (48, 48, 220),
        "bg": "#fff1f0",
        "border": "#cb3a31",
        "fg": "#6b140f",
    }


def banner_html(text: str, incident_type: str) -> str:
    theme = status_theme(incident_type)
    bg = theme["bg"]
    border = theme["border"]
    fg = theme["fg"]
    return (
        f"<div style='padding:0.85rem 1rem;border-radius:16px;border:2px solid {border};"
        f"background:{bg};color:{fg};font-weight:700;letter-spacing:0.02em;'>{text}</div>"
    )


def incident_panel(latest: Dict[str, Any]) -> str:
    return f"""
    <div style="padding:1rem;border-radius:18px;background:linear-gradient(145deg,#fffdf9,#f5f8fb);
    border:1px solid #d9e3ee;box-shadow:0 10px 24px rgba(13,38,59,0.07);">
      <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em;color:#5d6c7d;">
        Incident Summary
      </div>
      <div style="font-size:1.4rem;font-weight:800;color:#0f2233;margin-top:0.35rem;">
        {latest["incident_type"]}
      </div>
      <div style="margin-top:0.75rem;color:#203040;line-height:1.5;">
        {latest["message"]}
      </div>
      <div style="margin-top:0.8rem;font-size:0.86rem;color:#607182;text-transform:uppercase;letter-spacing:0.05em;">
        Recommended Response
      </div>
      <div style="margin-top:0.3rem;color:#102130;line-height:1.5;font-weight:600;">
        {latest["recommended_action"]}
      </div>
    </div>
    """


def metrics_panel(latest: Dict[str, Any]) -> str:
    return f"""
    <div style="padding:1rem;border-radius:18px;background:#0f1f2f;color:#f4f8fb;">
      <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.9rem;">
        <div>
          <div style="font-size:0.78rem;color:#92a7ba;text-transform:uppercase;letter-spacing:0.08em;">Gesture</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:0.2rem;">{latest["gesture"]}</div>
          <div style="font-size:0.86rem;color:#b9cad8;">Confidence {latest["gesture_confidence"]:.2f}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#92a7ba;text-transform:uppercase;letter-spacing:0.08em;">Distress</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:0.2rem;">{latest["distress_label"]}</div>
          <div style="font-size:0.86rem;color:#b9cad8;">Score {latest["distress_score"]:.2f}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#92a7ba;text-transform:uppercase;letter-spacing:0.08em;">Signal Lock</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:0.2rem;">{latest["stable_frames"]} frame(s)</div>
          <div style="font-size:0.86rem;color:#b9cad8;">{latest["active_signal"]}</div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#92a7ba;text-transform:uppercase;letter-spacing:0.08em;">Alerts Raised</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:0.2rem;">{latest["alert_count"]}</div>
          <div style="font-size:0.86rem;color:#b9cad8;">Severity {latest["severity"].upper()}</div>
        </div>
      </div>
    </div>
    """


def render_alert_history(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return """
        <div style="padding:1rem;border-radius:18px;border:1px dashed #c7d6e4;background:#fbfdff;color:#516170;">
          No incident has been raised yet. Hold one of the demo gestures steady for a few frames.
        </div>
        """

    cards = []
    for alert in alerts:
        colors = {
            "critical": ("#fff3ef", "#cf4b1f"),
            "high": ("#fff8ed", "#d38318"),
            "medium": ("#f0f8ff", "#3183c8"),
            "low": ("#eefaf1", "#29975b"),
        }
        bg, border = colors.get(alert["severity"], colors["low"])
        cards.append(
            f"""
            <div style="padding:0.9rem 1rem;border-radius:16px;border-left:6px solid {border};
            background:{bg};margin-bottom:0.75rem;">
              <div style="display:flex;justify-content:space-between;gap:1rem;">
                <div style="font-weight:800;color:#132433;">{alert["incident_type"]}</div>
                <div style="font-size:0.8rem;color:#546576;">{alert["created_at"]}</div>
              </div>
              <div style="margin-top:0.35rem;color:#243649;line-height:1.45;">{alert["message"]}</div>
              <div style="margin-top:0.45rem;font-size:0.86rem;color:#3f5162;">
                Gesture: {alert["gesture"]} | Distress: {alert["distress_label"]} ({alert["distress_score"]:.2f})
              </div>
            </div>
            """
        )
    return "".join(cards)


class VERSVideoProcessor(VideoProcessorBase):
    """WebRTC video processor for the VERS responder demo."""

    def __init__(
        self,
        stable_frames: int,
        cooldown_seconds: float,
        max_num_hands: int,
        mirror: bool,
    ) -> None:
        self.mirror = mirror
        self.gesture_recognizer = EmergencyGestureRecognizer()
        self.distress_analyzer = FaceDistressAnalyzer()
        self.engine = VERSAlertEngine(
            stable_frames=stable_frames,
            alert_cooldown_seconds=cooldown_seconds,
            log_path=ALERT_LOG_PATH,
        )

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.55,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )

        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self._hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self._face_tesselation = mp.solutions.face_mesh.FACEMESH_TESSELATION
        self._face_contours = mp.solutions.face_mesh.FACEMESH_CONTOURS
        self._lock = threading.Lock()
        self._latest = default_latest_state()
        self._frame_index = 0
        self._cached_face_landmarks = None
        self._cached_hand_landmarks = []
        self._cached_handedness = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._frame_index += 1
        frame_bgr = frame.to_ndarray(format="bgr24")
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        hand_frame = resize_for_inference(frame_bgr, max_side=720)
        face_frame = resize_for_inference(frame_bgr, max_side=640)
        if self._frame_index % 2 == 0 or not self._cached_hand_landmarks:
            hand_rgb = enhance_hand_detection_frame(hand_frame, upscale_limit=0)
            hand_rgb.flags.writeable = False
            hand_results = self.hands.process(hand_rgb)
            hand_rgb.flags.writeable = True
            self._cached_hand_landmarks = hand_results.multi_hand_landmarks or []
            self._cached_handedness = hand_results.multi_handedness or []

        if self._frame_index % 3 == 0 or self._cached_face_landmarks is None:
            face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_rgb.flags.writeable = False
            face_results = self.face_mesh.process(face_rgb)
            face_rgb.flags.writeable = True
            self._cached_face_landmarks = (
                face_results.multi_face_landmarks[0]
                if face_results.multi_face_landmarks
                else None
            )

        gesture_candidates = []
        if self._cached_hand_landmarks:
            for index, hand_landmarks in enumerate(self._cached_hand_landmarks):
                handedness = None
                if self._cached_handedness and index < len(self._cached_handedness):
                    handedness = self._cached_handedness[index].classification[0].label
                gesture_candidates.append(self.gesture_recognizer.detect(hand_landmarks, handedness))
                self._drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._hand_connections,
                    self._drawing.DrawingSpec(color=(34, 204, 127), thickness=2, circle_radius=1),
                    self._drawing.DrawingSpec(color=(10, 120, 90), thickness=2, circle_radius=1),
                )

        distress = calm_distress()
        if self._cached_face_landmarks is not None:
            face_landmarks = self._cached_face_landmarks
            self._drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self._face_contours,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._drawing_styles.get_default_face_mesh_contours_style(),
            )
            distress = self.distress_analyzer.analyze(face_landmarks)
            for landmark_index in (13, 14, 159, 145, 386, 374):
                point = face_landmarks.landmark[landmark_index]
                x = int(point.x * frame_bgr.shape[1])
                y = int(point.y * frame_bgr.shape[0])
                cv2.circle(frame_bgr, (x, y), 3, (255, 200, 0), -1)

        latest = self.engine.update(gesture_candidates=gesture_candidates, distress=distress, now=time.time())
        latest_dict = asdict(latest)
        self._overlay(frame_bgr, latest=latest_dict)

        with self._lock:
            self._latest = latest_dict

        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

    def _overlay(self, frame_bgr: Any, latest: Dict[str, Any]) -> None:
        color = status_theme(latest["incident_type"])["overlay_bgr"]
        cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 70), color, -1)
        cv2.putText(
            frame_bgr,
            latest["status_banner"],
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Gesture: {latest['gesture']} ({latest['gesture_confidence']:.2f})",
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Distress: {latest['distress_label']} ({latest['distress_score']:.2f})",
            (18, frame_bgr.shape[0] - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Incident: {latest['incident_type']}",
            (18, frame_bgr.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def __del__(self) -> None:
        try:
            self.hands.close()
            self.face_mesh.close()
        except Exception:
            pass


def add_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(255,209,163,0.38), transparent 28%),
            linear-gradient(180deg, #fff9f2 0%, #eef4fb 55%, #f7fbff 100%);
        }
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="VERS Demo Console", layout="wide")
    add_page_style()

    st.title("Vision-Based Emergency Response System")
    st.caption(
        "Laptop-webcam demo for silent emergency gestures, facial distress scoring, and responder alerts."
    )

    with st.sidebar:
        st.header("Demo Controls")
        video_quality = st.selectbox(
            "Video quality",
            options=list(VIDEO_PRESETS.keys()),
            index=1,
            help="Higher quality looks better but uses more webcam and browser processing power.",
        )
        stable_frames = st.slider("Stable frames before locking signal", min_value=3, max_value=10, value=6)
        cooldown_seconds = st.slider(
            "Alert cooldown (seconds)",
            min_value=3,
            max_value=20,
            value=8,
        )
        max_num_hands = st.selectbox("Maximum hands to track", options=[1, 2], index=1)
        mirror = st.toggle("Mirror webcam preview", value=True)
        st.markdown("**Gesture vocabulary**")
        st.markdown(
            "\n".join(
                [
                    "- `Open palm` -> SOS / general emergency",
                    "- `Closed fist` -> Medical emergency",
                    "- `V sign` -> Accident / injury",
                    "- `Index finger up` -> Security / police help",
                    "- `Thumbs up` -> Safe / resolved",
                ]
            )
        )
        st.info(f"Incident log is saved to `{ALERT_LOG_PATH}`")
        preset = VIDEO_PRESETS[video_quality]
        st.caption(
            f"Requested stream: {preset['width']}x{preset['height']} at {preset['frame_rate']} FPS"
        )

    top_left, top_right = st.columns([1.25, 1.0], gap="large")
    with top_left:
        status_box = st.empty()
        webcam_box = st.container()
    with top_right:
        incident_box = st.empty()
        metrics_box = st.empty()

    st.markdown("### Alert Timeline")
    history_box = st.empty()
    st.markdown("### Demo Notes")
    st.markdown(
        "\n".join(
            [
                "- Hold a gesture for a few frames so the system can stabilize it before raising an alert.",
                "- Facial distress can escalate the severity even when the gesture stays the same.",
                "- The app writes every dispatched event to a local JSONL log for demo evidence.",
            ]
        )
    )

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    preset = VIDEO_PRESETS[video_quality]

    with webcam_box:
        ctx = webrtc_streamer(
            key="vers-demo-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints=build_media_constraints(
                width=preset["width"],
                height=preset["height"],
                frame_rate=preset["frame_rate"],
            ),
            async_processing=True,
            video_processor_factory=lambda: VERSVideoProcessor(
                stable_frames=stable_frames,
                cooldown_seconds=float(cooldown_seconds),
                max_num_hands=int(max_num_hands),
                mirror=mirror,
            ),
        )

    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                latest = ctx.video_processor.get_latest()
                status_box.markdown(
                    banner_html(latest["status_banner"], latest["incident_type"]),
                    unsafe_allow_html=True,
                )
                incident_box.markdown(incident_panel(latest), unsafe_allow_html=True)
                metrics_box.markdown(metrics_panel(latest), unsafe_allow_html=True)
                history_box.markdown(render_alert_history(latest["alerts"]), unsafe_allow_html=True)
            time.sleep(0.25)
    else:
        status_box.markdown(
            banner_html("LIVE MONITORING: CAMERA NOT STARTED", "Monitoring"),
            unsafe_allow_html=True,
        )
        incident_box.markdown(
            """
            <div style="padding:1rem;border-radius:18px;background:#ffffff;border:1px solid #d9e3ee;">
              Press <strong>START</strong> on the webcam widget to begin the live demo.
            </div>
            """,
            unsafe_allow_html=True,
        )
        metrics_box.markdown(metrics_panel(default_latest_state()), unsafe_allow_html=True)
        history_box.markdown(render_alert_history([]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
