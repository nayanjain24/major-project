"""Real-time gesture inference from webcam frames.

Pipeline:
1. Capture webcam frames via OpenCV.
2. Extract pose + hand keypoints using MediaPipe.
3. Maintain a fixed-size sliding window of keypoints directly on target device.
4. Run a trained LSTM gesture model on the window (optionally every N frames).
5. Run facial emotion detection using DeepFace.
6. Fuse gesture + emotion with simple rules into context-aware text.
7. Display live predictions in real time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from models import GestureLSTMClassifier
from utils import FacialEmotionDetector


class SlidingTensorWindow:
    """Low-overhead fixed-size sliding window stored as a device tensor."""

    def __init__(self, seq_len: int, input_dim: int, device: torch.device) -> None:
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.device = device
        self.buffer = torch.zeros((seq_len, input_dim), dtype=torch.float32, device=device)
        self.count = 0
        self.write_idx = 0

    def append(self, keypoints: np.ndarray) -> None:
        """Append a frame vector and keep only the most recent `seq_len` frames."""
        if keypoints.shape != (self.input_dim,):
            raise ValueError(
                f"Expected keypoint shape ({self.input_dim},), got {tuple(keypoints.shape)}"
            )

        row = torch.from_numpy(keypoints)
        if self.device.type != "cpu":
            row = row.to(self.device, non_blocking=True)

        # Ring buffer write: O(input_dim) per frame without shifting entire window.
        self.buffer[self.write_idx].copy_(row)
        self.write_idx = (self.write_idx + 1) % self.seq_len
        if self.count < self.seq_len:
            self.count += 1

    def is_ready(self) -> bool:
        return self.count == self.seq_len

    def batch(self) -> torch.Tensor:
        """Return model input shaped [1, seq_len, input_dim]."""
        if self.count < self.seq_len:
            return self.buffer[: self.count].unsqueeze(0)

        # Reorder ring buffer into temporal order: oldest -> newest.
        if self.write_idx == 0:
            ordered = self.buffer
        else:
            ordered = torch.cat(
                [self.buffer[self.write_idx :], self.buffer[: self.write_idx]], dim=0
            )
        return ordered.unsqueeze(0)

    def filled(self) -> int:
        return self.count


def choose_device() -> torch.device:
    """Pick CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_keypoints(pose_results: Any, hand_results: Any, input_dim: int = 258) -> np.ndarray:
    """Convert MediaPipe landmarks to a fixed-size keypoint vector.

    Default layout (input_dim=258):
    - Pose: 33 * (x, y, z, visibility) = 132
    - Left hand: 21 * (x, y, z) = 63
    - Right hand: 21 * (x, y, z) = 63
    """
    if input_dim != 258:
        raise ValueError(
            "This extractor currently supports input_dim=258. "
            f"Got input_dim={input_dim}."
        )

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


def draw_skeleton_overlay(frame: np.ndarray, pose_results: Any, hand_results: Any) -> None:
    """Draw pose + hand landmarks on the frame."""
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    if pose_results.pose_landmarks:
        drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )

    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_lms,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=styles.get_default_hand_connections_style(),
            )


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[GestureLSTMClassifier, Dict[int, str], int, int]:
    """Restore model, label map, sequence length, and input dim from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    input_dim = int(config.get("input_dim", 258))
    hidden_dim = int(config.get("hidden_dim", 256))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.3))
    bidirectional = bool(config.get("bidirectional", False))
    seq_len = int(config.get("seq_len", 30))

    label_to_index = checkpoint.get("label_to_index", {})
    if not label_to_index:
        raise ValueError("Checkpoint is missing label_to_index mapping.")
    num_classes = len(label_to_index)

    model = GestureLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Normalize possible int/string key variations after checkpoint load.
    raw_index_to_label = checkpoint.get("index_to_label")
    if isinstance(raw_index_to_label, dict) and raw_index_to_label:
        index_to_label = {int(k): str(v) for k, v in raw_index_to_label.items()}
    else:
        index_to_label = {int(v): str(k) for k, v in label_to_index.items()}

    return model, index_to_label, seq_len, input_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time gesture inference from webcam.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to trained model checkpoint file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Optional override for sequence window size. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional minimum confidence to display class label.",
    )
    parser.add_argument(
        "--emotion-interval",
        type=int,
        default=3,
        help="Run emotion detection every N frames to reduce latency.",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="Run gesture model every N frames (prediction is reused between runs).",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=1.0,
        help="Downscale frame for MediaPipe/emotion processing (0.25-1.0).",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=0,
        help="MediaPipe Pose complexity (0 is fastest).",
    )
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=2,
        help="Maximum hands to track; lower values can reduce compute.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Requested webcam width.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=480,
        help="Requested webcam height.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable skeleton drawing to reduce CPU load.",
    )
    return parser.parse_args()


def _to_probability(score: float) -> float:
    """Normalize confidence to [0, 1], supporting both 0-1 and 0-100 scales."""
    score = float(score)
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _emotion_context(emotion_label: Optional[str], emotion_confidence: float) -> str:
    """Map raw emotion labels into coarse context tags."""
    emotion = (emotion_label or "").strip().lower()
    conf = _to_probability(emotion_confidence)

    if conf < 0.35:
        return "neutral"
    if emotion in {"urgent", "fear", "angry", "sad", "disgust"}:
        return "urgent"
    if emotion in {"surprise"}:
        return "alert"
    if emotion in {"happy"}:
        return "positive"
    return "neutral"


def fuse_predictions(
    gesture_label: str,
    gesture_confidence: float,
    emotion_label: Optional[str],
    emotion_confidence: float,
) -> str:
    """Combine gesture and emotion into context-aware output text."""
    gesture = (gesture_label or "").strip().lower()
    gesture_conf = _to_probability(gesture_confidence)
    context = _emotion_context(emotion_label, emotion_confidence)

    if not gesture or gesture in {"collecting sequence...", "uncertain"} or gesture_conf < 0.2:
        return "COLLECTING SIGNAL..."

    # Special-case safety/emergency gestures.
    if gesture in {"help", "emergency", "danger", "sos"} and context == "urgent":
        return "URGENT HELP NEEDED"

    text = gesture.replace("_", " ").upper()
    if context == "urgent":
        return f"URGENT: {text}"
    if context == "alert":
        return f"ALERT: {text}"
    if context == "positive" and gesture in {"thanks", "thank_you", "thankyou"}:
        return "APPRECIATIVE THANK YOU"
    return text


def main() -> None:
    args = parse_args()
    device = choose_device()

    model, index_to_label, checkpoint_seq_len, input_dim = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    seq_len = int(args.seq_len) if args.seq_len is not None else checkpoint_seq_len
    if seq_len <= 0:
        raise ValueError("--seq-len must be > 0")
    if args.emotion_interval <= 0:
        raise ValueError("--emotion-interval must be > 0")
    if args.inference_interval <= 0:
        raise ValueError("--inference-interval must be > 0")
    if args.process_scale <= 0.0 or args.process_scale > 1.0:
        raise ValueError("--process-scale must be in (0, 1].")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    emotion_detector = FacialEmotionDetector()
    window = SlidingTensorWindow(seq_len=seq_len, input_dim=input_dim, device=device)

    current_label = "Collecting sequence..."
    current_confidence = 0.0
    current_emotion_label = "unknown"
    current_emotion_confidence = 0.0
    current_emotion_bbox: Optional[Dict[str, int]] = None
    fused_output = "COLLECTING SIGNAL..."
    frame_count = 0
    last_tick = time.perf_counter()
    smoothed_fps = 0.0

    print(f"Device: {device}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Sliding window size: {seq_len}")
    print(f"Emotion detection interval: every {args.emotion_interval} frame(s)")
    print(f"Gesture inference interval: every {args.inference_interval} frame(s)")
    print(f"Processing scale: {args.process_scale}")
    print("Press 'q' to quit.")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed; exiting.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            if args.process_scale < 1.0:
                process_frame = cv2.resize(
                    frame,
                    None,
                    fx=args.process_scale,
                    fy=args.process_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                process_frame = frame

            rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_results = pose.process(rgb)
            hand_results = hands.process(rgb)
            rgb.flags.writeable = True

            if not args.no_overlay:
                draw_skeleton_overlay(frame, pose_results, hand_results)

            # Emotion detection can be expensive; run every N frames.
            if frame_count % args.emotion_interval == 0:
                emotion_result = emotion_detector.detect(process_frame)
                if emotion_result["face_detected"]:
                    current_emotion_label = str(emotion_result.get("emotion") or "unknown")
                    current_emotion_confidence = float(emotion_result.get("confidence", 0.0))
                    bbox = emotion_result.get("bbox")
                    if bbox and args.process_scale < 1.0:
                        inv = 1.0 / args.process_scale
                        current_emotion_bbox = {
                            "x": int(bbox["x"] * inv),
                            "y": int(bbox["y"] * inv),
                            "w": int(bbox["w"] * inv),
                            "h": int(bbox["h"] * inv),
                        }
                    else:
                        current_emotion_bbox = bbox
                else:
                    current_emotion_label = "unknown"
                    current_emotion_confidence = 0.0
                    current_emotion_bbox = None

            keypoints = extract_keypoints(
                pose_results=pose_results,
                hand_results=hand_results,
                input_dim=input_dim,
            )
            window.append(keypoints=keypoints)

            # Running the model less frequently lowers latency on slower hardware.
            if window.is_ready() and (frame_count % args.inference_interval == 0):
                probs = model(window.batch())[0]
                conf, pred_idx = torch.max(probs, dim=0)
                pred_conf = float(conf.item())
                pred_idx_int = int(pred_idx.item())

                if pred_conf >= args.confidence_threshold:
                    current_label = index_to_label.get(pred_idx_int, f"class_{pred_idx_int}")
                    current_confidence = pred_conf
                else:
                    current_label = "uncertain"
                    current_confidence = pred_conf

            fused_output = fuse_predictions(
                gesture_label=current_label,
                gesture_confidence=current_confidence,
                emotion_label=current_emotion_label,
                emotion_confidence=current_emotion_confidence,
            )

            if current_emotion_bbox is not None:
                x = int(current_emotion_bbox["x"])
                y = int(current_emotion_bbox["y"])
                w = int(current_emotion_bbox["w"])
                h = int(current_emotion_bbox["h"])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 140, 0), 2)

            cv2.putText(
                frame,
                f"Prediction: {current_label}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Confidence: {current_confidence:.2f}",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Emotion: {current_emotion_label} ({_to_probability(current_emotion_confidence):.2f})",
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Fused: {fused_output}",
                (15, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            now = time.perf_counter()
            dt = max(now - last_tick, 1e-6)
            last_tick = now
            fps = 1.0 / dt
            smoothed_fps = fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9 + fps * 0.1)
            cv2.putText(
                frame,
                f"Window: {window.filled()}/{seq_len}",
                (15, 164),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (15, 196),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-time Gesture Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
