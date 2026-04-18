"""Capture a gesture dataset using MediaPipe Pose + Hands.

Features:
- Shows live skeleton overlays (pose + both hands).
- Records keypoints over time from webcam frames.
- Saves each recording as a NumPy array (.npy).
- Organizes saved recordings in per-gesture folders.
- Lets you label/relabel recordings by gesture name.

Keyboard controls:
- r: toggle recording on/off
- s: save current recorded sequence
- c: clear current sequence buffer
- l: change gesture label
- q: quit
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np


def sanitize_label(label: str) -> str:
    """Normalize a gesture label so it is safe as a folder name."""
    cleaned = label.strip().lower().replace(" ", "_")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"})


def get_label_from_user(default: str | None = None) -> str:
    """Prompt for a gesture label in the terminal."""
    while True:
        prompt = "Enter gesture label"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        value = input(prompt).strip()
        if not value and default:
            value = default

        value = sanitize_label(value)
        if value:
            return value

        print("Label cannot be empty. Try again.")


def extract_keypoints(
    pose_results: mp.solutions.pose.Pose,
    hand_results: mp.solutions.hands.Hands,
) -> np.ndarray:
    """Create a fixed-size feature vector from pose and hand landmarks.

    Vector layout:
    - Pose: 33 landmarks * (x, y, z, visibility) = 132 values
    - Left hand: 21 landmarks * (x, y, z) = 63 values
    - Right hand: 21 landmarks * (x, y, z) = 63 values
    Total = 258 float values per frame.
    """

    # Pose keypoints (or zeros when not detected).
    if pose_results.pose_landmarks:
        pose = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # Hands keypoints keyed by handedness label for deterministic order.
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lms, handedness in zip(
            hand_results.multi_hand_landmarks, hand_results.multi_handedness
        ):
            label = handedness.classification[0].label.lower()  # "left" or "right"
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            ).flatten()
            if label == "left":
                left_hand = coords
            elif label == "right":
                right_hand = coords

    return np.concatenate([pose, left_hand, right_hand], axis=0)


def draw_landmarks(
    frame: np.ndarray,
    pose_results: mp.solutions.pose.Pose,
    hand_results: mp.solutions.hands.Hands,
) -> None:
    """Draw pose and hand skeleton overlays onto the frame."""
    drawing = mp.solutions.drawing_utils
    pose_styles = mp.solutions.drawing_styles
    hands_styles = mp.solutions.drawing_styles

    if pose_results.pose_landmarks:
        drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_styles.get_default_pose_landmarks_style(),
        )

    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_lms,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hands_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=hands_styles.get_default_hand_connections_style(),
            )


def save_sequence(sequence: List[np.ndarray], output_dir: Path, gesture_label: str) -> Path:
    """Save recorded keypoint sequence to a gesture-specific folder."""
    gesture_dir = output_dir / gesture_label
    gesture_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = gesture_dir / f"{gesture_label}_{timestamp}_{len(sequence)}f.npy"

    sequence_array = np.stack(sequence).astype(np.float32)  # shape: [T, 258]
    np.save(target_path, sequence_array)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture gesture keypoint sequences using MediaPipe Pose and Hands."
    )
    parser.add_argument(
        "--gesture",
        type=str,
        default=None,
        help="Initial gesture label. If omitted, you'll be prompted.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Directory where per-gesture folders and .npy sequences will be saved.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum frames required before saving a sequence.",
    )
    args = parser.parse_args()

    gesture_label = sanitize_label(args.gesture) if args.gesture else None
    if not gesture_label:
        gesture_label = get_label_from_user(default="gesture")

    args.data_dir.mkdir(parents=True, exist_ok=True)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera-index and camera permissions.")

    recording = False
    sequence: List[np.ndarray] = []

    print(
        "\nControls: [r] record toggle | [s] save | [c] clear | [l] relabel | [q] quit\n"
    )

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed; stopping.")
                break

            frame = cv2.flip(frame, 1)  # Mirror view for easier interaction.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)

            rgb.flags.writeable = True
            draw_landmarks(frame, pose_results, hand_results)

            if recording:
                keypoints = extract_keypoints(pose_results, hand_results)
                sequence.append(keypoints)

            status = "REC" if recording else "IDLE"
            cv2.putText(
                frame,
                f"Gesture: {gesture_label}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Mode: {status} | Frames: {len(sequence)}",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if recording else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "r:record  s:save  c:clear  l:label  q:quit",
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gesture Dataset Capture (Pose + Hands)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                recording = not recording
                print(f"Recording {'started' if recording else 'paused'}")
            elif key == ord("c"):
                sequence.clear()
                print("Current sequence cleared.")
            elif key == ord("l"):
                recording = False
                gesture_label = get_label_from_user(default=gesture_label)
                print(f"Gesture label set to: {gesture_label}")
            elif key == ord("s"):
                if len(sequence) < args.min_frames:
                    print(
                        f"Not saved: need at least {args.min_frames} frames (have {len(sequence)})."
                    )
                else:
                    saved_path = save_sequence(sequence, args.data_dir, gesture_label)
                    print(f"Saved: {saved_path}")
                    sequence.clear()
                    recording = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
