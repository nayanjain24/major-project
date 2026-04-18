"""Unit tests for the rule-based VERS demo engine."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from app.vers_engine import (
    DistressReading,
    EmergencyGestureRecognizer,
    VERSAlertEngine,
    calm_distress,
)


def make_hand(points: list[tuple[float, float]]) -> SimpleNamespace:
    landmarks = [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]
    return SimpleNamespace(landmark=landmarks)


class GestureRecognizerTests(unittest.TestCase):
    def test_open_palm_maps_to_sos(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        reading = recognizer.recognize_finger_states([1, 1, 1, 1, 1], margins=[0.2] * 5)
        self.assertEqual(reading.label, "SOS")
        self.assertGreaterEqual(reading.confidence, 0.8)

    def test_v_sign_maps_to_accident(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        reading = recognizer.recognize_finger_states([0, 1, 1, 0, 0], margins=[0.18] * 5)
        self.assertEqual(reading.label, "ACCIDENT")
        self.assertEqual(reading.incident_type, "Accident / injury")

    def test_detect_thumbs_up_from_landmarks(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        hand = make_hand(
            [
                (0.50, 0.86),
                (0.47, 0.76),
                (0.46, 0.66),
                (0.45, 0.56),
                (0.44, 0.42),
                (0.43, 0.69),
                (0.46, 0.79),
                (0.47, 0.83),
                (0.48, 0.85),
                (0.50, 0.67),
                (0.51, 0.78),
                (0.515, 0.82),
                (0.52, 0.85),
                (0.56, 0.68),
                (0.56, 0.79),
                (0.555, 0.825),
                (0.55, 0.85),
                (0.61, 0.71),
                (0.595, 0.80),
                (0.58, 0.835),
                (0.565, 0.85),
            ]
        )
        reading = recognizer.detect(hand, "Right")
        self.assertEqual(reading.label, "SAFE")
        self.assertGreaterEqual(reading.confidence, 0.85)

    def test_detect_fist_from_landmarks(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        hand = make_hand(
            [
                (0.50, 0.86),
                (0.47, 0.77),
                (0.455, 0.72),
                (0.445, 0.695),
                (0.435, 0.69),
                (0.43, 0.69),
                (0.46, 0.75),
                (0.475, 0.77),
                (0.49, 0.785),
                (0.50, 0.67),
                (0.51, 0.74),
                (0.515, 0.765),
                (0.52, 0.785),
                (0.56, 0.68),
                (0.555, 0.745),
                (0.55, 0.77),
                (0.545, 0.79),
                (0.61, 0.71),
                (0.59, 0.76),
                (0.575, 0.782),
                (0.56, 0.80),
            ]
        )
        reading = recognizer.detect(hand, "Right")
        self.assertEqual(reading.label, "MEDICAL")
        self.assertGreaterEqual(reading.confidence, 0.85)


class AlertEngineTests(unittest.TestCase):
    def test_medical_gesture_raises_alert_after_stability(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        medical = recognizer.recognize_finger_states([0, 0, 0, 0, 0], margins=[0.18] * 5)
        engine = VERSAlertEngine(stable_frames=3, alert_cooldown_seconds=3.0)

        latest = None
        for frame_idx in range(3):
            latest = engine.update([medical], calm_distress(), now=float(frame_idx))

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.incident_type, "Medical emergency")
        self.assertEqual(latest.alert_count, 1)
        self.assertIsNotNone(latest.new_alert)

    def test_safe_signal_stays_non_alerting(self) -> None:
        recognizer = EmergencyGestureRecognizer()
        safe = recognizer.recognize_finger_states([1, 0, 0, 0, 0], margins=[0.2] * 5)
        engine = VERSAlertEngine(stable_frames=3, alert_cooldown_seconds=3.0)

        latest = None
        for frame_idx in range(4):
            latest = engine.update([safe], calm_distress(), now=float(frame_idx))

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.incident_type, "Safe / resolved")
        self.assertEqual(latest.alert_count, 0)

    def test_distress_only_can_raise_silent_alert(self) -> None:
        engine = VERSAlertEngine(stable_frames=3, alert_cooldown_seconds=3.0)
        severe_distress = DistressReading(
            label="CRITICAL DISTRESS",
            score=0.9,
            mouth_ratio=0.12,
            eye_ratio=0.3,
            brow_ratio=0.28,
        )

        latest = None
        for frame_idx in range(3):
            latest = engine.update([], severe_distress, now=float(frame_idx))

        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.incident_type, "Silent distress")
        self.assertEqual(latest.severity, "critical")
        self.assertEqual(latest.alert_count, 1)


if __name__ == "__main__":
    unittest.main()
