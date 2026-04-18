"""MediaPipe-based landmark tracker.

This module handles real-time landmark extraction from camera frames.
In a sign-language system, these landmarks become features for the ML model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import mediapipe as mp


class MediaPipeTracker:
    """Wrapper around MediaPipe Holistic for body/hand/face landmarks."""

    def __init__(self, static_image_mode: bool = False) -> None:
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
        )

    def process_frame(self, frame_bgr: Any) -> Optional[Dict[str, Any]]:
        """Extract landmarks from a BGR frame.

        Args:
            frame_bgr: OpenCV frame in BGR format.

        Returns:
            Dictionary of landmark results, or `None` when no frame is provided.
        """
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(frame_rgb)

        return {
            "pose_landmarks": results.pose_landmarks,
            "left_hand_landmarks": results.left_hand_landmarks,
            "right_hand_landmarks": results.right_hand_landmarks,
            "face_landmarks": results.face_landmarks,
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._holistic.close()
