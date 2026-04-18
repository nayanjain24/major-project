"""Facial emotion detection for webcam frames using DeepFace.

This module is designed for real-time loops:
- detect face region with OpenCV Haar cascade
- run DeepFace emotion prediction on the cropped face
- return a compact dictionary for downstream app logic
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
try:
    from deepface import DeepFace
except Exception:  # pragma: no cover - optional dependency import path
    DeepFace = None


class FacialEmotionDetector:
    """Detect face region and estimate emotion label/confidence."""

    def __init__(self, min_face_size: Tuple[int, int] = (40, 40)) -> None:
        # Haar cascade is lightweight and fast enough for frame-by-frame use.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        self._min_face_size = min_face_size
        self._deepface_available = DeepFace is not None

    def detect(self, frame_bgr: Any) -> Dict[str, Any]:
        """Analyze a single webcam frame and return emotion results.

        Args:
            frame_bgr: OpenCV frame in BGR format.

        Returns:
            Dictionary format:
            {
                "face_detected": bool,
                "bbox": {"x": int, "y": int, "w": int, "h": int} | None,
                "emotion": str | None,
                "confidence": float,
                "error": str | None,
            }
        """
        if frame_bgr is None:
            return self._empty_result(error="Input frame is None.")
        if not self._deepface_available:
            return self._empty_result(
                error="DeepFace is not available. Install dependencies from requirements.txt."
            )

        bbox = self._detect_face(frame_bgr)
        if bbox is None:
            return self._empty_result(error="No face detected in frame.")

        x, y, w, h = bbox
        face_roi = frame_bgr[y : y + h, x : x + w]
        if face_roi.size == 0:
            return self._empty_result(error="Detected face crop is empty.")

        try:
            # We pass a cropped face, so enforce_detection can be disabled for stability.
            analysis = DeepFace.analyze(
                img_path=face_roi,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:  # pragma: no cover - runtime/library-dependent branch
            return self._empty_result(error=f"DeepFace analysis failed: {exc}")

        result = analysis[0] if isinstance(analysis, list) else analysis
        emotion_label = result.get("dominant_emotion")
        confidence = self._extract_confidence(result, emotion_label)

        return {
            "face_detected": True,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "emotion": emotion_label,
            "confidence": float(confidence),
            "error": None,
        }

    def _detect_face(self, frame_bgr: Any) -> Optional[Tuple[int, int, int, int]]:
        """Find the most prominent face and return (x, y, w, h)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self._min_face_size,
        )

        if len(faces) == 0:
            return None

        # Use the largest face in frame (common assumption for webcam interaction).
        return max(faces, key=lambda box: box[2] * box[3])

    @staticmethod
    def _extract_confidence(result: Dict[str, Any], emotion_label: Optional[str]) -> float:
        """Extract confidence for the dominant emotion from DeepFace output."""
        if not emotion_label:
            return 0.0
        emotions = result.get("emotion")
        if not isinstance(emotions, dict):
            return 0.0
        value = emotions.get(emotion_label, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _empty_result(error: str) -> Dict[str, Any]:
        """Return a consistent empty response shape."""
        return {
            "face_detected": False,
            "bbox": None,
            "emotion": None,
            "confidence": 0.0,
            "error": error,
        }
