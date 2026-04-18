"""OpenCV fallback demo for the Vision-Based Emergency Response System (VERS).

Run:
    python -m app.vers_demo
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import platform
import time

import cv2
import mediapipe as mp

from app.vers_engine import (
    EmergencyGestureRecognizer,
    FaceDistressAnalyzer,
    VERSAlertEngine,
    calm_distress,
    enhance_hand_detection_frame,
    resize_for_inference,
)


ALERT_LOG_PATH = Path("runtime/vers_alerts_terminal.jsonl")


def is_macos() -> bool:
    return platform.system() == "Darwin"


def init_camera() -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if is_macos() else cv2.CAP_ANY
    capture = cv2.VideoCapture(0, backend)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_FPS, 24 if is_macos() else 24)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def severity_color(severity: str) -> tuple[int, int, int]:
    palette = {
        "critical": (24, 24, 210),
        "high": (14, 116, 255),
        "medium": (0, 190, 245),
        "low": (44, 190, 100),
    }
    return palette.get(severity, (150, 150, 150))


def run_demo() -> None:
    gesture_recognizer = EmergencyGestureRecognizer()
    distress_analyzer = FaceDistressAnalyzer()
    engine = VERSAlertEngine(
        stable_frames=6,
        alert_cooldown_seconds=8.0,
        log_path=ALERT_LOG_PATH,
    )

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.55,
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    cap = init_camera()

    if not cap.isOpened():
        raise RuntimeError(
            "Could not access camera index 0. On macOS, enable camera permissions for Terminal."
        )

    print("VERS terminal demo started.")
    print("Gestures: open palm=S.O.S | fist=medical | V sign=accident | index up=security")
    print("Press 'q' to stop.\n")

    last_alert_id = None
    frame_index = 0
    cached_face_landmarks = None
    cached_hand_landmarks = []
    cached_handedness = []

    try:
        while True:
            frame_index += 1
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            hand_frame = resize_for_inference(frame, max_side=720)
            face_frame = resize_for_inference(frame, max_side=640)
            if frame_index % 2 == 0 or not cached_hand_landmarks:
                hand_rgb = enhance_hand_detection_frame(hand_frame, upscale_limit=0)
                hand_rgb.flags.writeable = False
                hand_results = hands.process(hand_rgb)
                hand_rgb.flags.writeable = True
                cached_hand_landmarks = hand_results.multi_hand_landmarks or []
                cached_handedness = hand_results.multi_handedness or []

            if frame_index % 3 == 0 or cached_face_landmarks is None:
                face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_rgb.flags.writeable = False
                face_results = face_mesh.process(face_rgb)
                face_rgb.flags.writeable = True
                cached_face_landmarks = (
                    face_results.multi_face_landmarks[0]
                    if face_results.multi_face_landmarks
                    else None
                )

            gesture_candidates = []
            if cached_hand_landmarks:
                for index, hand_landmarks in enumerate(cached_hand_landmarks):
                    handedness = None
                    if cached_handedness and index < len(cached_handedness):
                        handedness = cached_handedness[index].classification[0].label
                    gesture_candidates.append(gesture_recognizer.detect(hand_landmarks, handedness))
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            distress = calm_distress()
            if cached_face_landmarks is not None:
                face_landmarks = cached_face_landmarks
                drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                )
                distress = distress_analyzer.analyze(face_landmarks)

            latest = engine.update(
                gesture_candidates=gesture_candidates,
                distress=distress,
                now=time.time(),
            )
            latest_dict = asdict(latest)

            color = severity_color(latest.severity)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), color, -1)
            cv2.putText(
                frame,
                latest.status_banner,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Gesture: {latest.gesture} ({latest.gesture_confidence:.2f})",
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Distress: {latest.distress_label} ({latest.distress_score:.2f})",
                (16, frame.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Action: {latest.recommended_action[:70]}",
                (16, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if latest.new_alert:
                alert_id = latest.new_alert["alert_id"]
                if alert_id != last_alert_id:
                    print(f"[{latest.new_alert['created_at']}] {latest.new_alert['severity'].upper()} | "
                          f"{latest.new_alert['incident_type']} | {latest.new_alert['message']}")
                    last_alert_id = alert_id

            cv2.imshow("VERS Emergency Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        print(f"\nClosed VERS demo. Alert log: {ALERT_LOG_PATH}")


if __name__ == "__main__":
    run_demo()
