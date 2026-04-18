"""Core webcam demo logic for the Vision-Based Emergency Response System (VERS).

This module keeps the demo self-contained and laptop-friendly:
- rule-based emergency gesture recognition from MediaPipe hand landmarks
- face-landmark distress scoring without requiring a trained model checkpoint
- temporal smoothing to avoid noisy frame-by-frame outputs
- structured alert generation and JSONL incident logging
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

import cv2


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _scale(value: float, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    return _clamp((value - start) / (end - start))


def _distance(point_a: Any, point_b: Any) -> float:
    dx = float(point_a.x) - float(point_b.x)
    dy = float(point_a.y) - float(point_b.y)
    dz = float(getattr(point_a, "z", 0.0)) - float(getattr(point_b, "z", 0.0))
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _point(x: float, y: float, z: float = 0.0) -> Any:
    return type("Point", (), {"x": x, "y": y, "z": z})()


def _average_point(points: Sequence[Any]) -> Any:
    count = max(len(points), 1)
    return _point(
        sum(float(point.x) for point in points) / count,
        sum(float(point.y) for point in points) / count,
        sum(float(getattr(point, "z", 0.0)) for point in points) / count,
    )


def enhance_hand_detection_frame(frame_bgr: Any, upscale_limit: int = 960) -> Any:
    """Improve webcam frames before MediaPipe Hands inference."""
    frame = frame_bgr
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    if longest_side < upscale_limit:
        scale = upscale_limit / float(longest_side)
        frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.18, blurred, -0.18, 0.0)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)


def resize_for_inference(frame_bgr: Any, max_side: int) -> Any:
    """Resize a frame so the longest side matches `max_side` at most."""
    height, width = frame_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return frame_bgr
    scale = max_side / float(longest_side)
    return cv2.resize(
        frame_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


@dataclass(frozen=True)
class GestureDefinition:
    label: str
    pattern: tuple[int, int, int, int, int]
    incident_type: str
    default_message: str
    recommended_action: str
    severity_hint: str


@dataclass
class GestureReading:
    label: str
    confidence: float
    incident_type: str
    default_message: str
    recommended_action: str
    severity_hint: str
    finger_states: tuple[int, int, int, int, int]


@dataclass
class DistressReading:
    label: str
    score: float
    mouth_ratio: float
    eye_ratio: float
    brow_ratio: float


@dataclass
class AlertRecord:
    alert_id: str
    created_at: str
    incident_type: str
    severity: str
    gesture: str
    gesture_confidence: float
    distress_label: str
    distress_score: float
    message: str
    recommended_action: str
    source: str = "webcam"
    status: str = "queued"


@dataclass
class LiveStatus:
    gesture: str
    gesture_confidence: float
    distress_label: str
    distress_score: float
    incident_type: str
    severity: str
    status_banner: str
    message: str
    recommended_action: str
    stable_frames: int
    active_signal: str
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    alert_count: int = 0
    new_alert: Optional[Dict[str, Any]] = None
    log_path: Optional[str] = None


EMERGENCY_GESTURES: tuple[GestureDefinition, ...] = (
    GestureDefinition(
        label="SOS",
        pattern=(1, 1, 1, 1, 1),
        incident_type="General emergency",
        default_message="General emergency signal received from webcam station.",
        recommended_action="Dispatch the nearest available response team immediately.",
        severity_hint="critical",
    ),
    GestureDefinition(
        label="MEDICAL",
        pattern=(0, 0, 0, 0, 0),
        incident_type="Medical emergency",
        default_message="Possible medical emergency or collapse reported by hand signal.",
        recommended_action="Notify medical responders and keep a priority line open.",
        severity_hint="critical",
    ),
    GestureDefinition(
        label="ACCIDENT",
        pattern=(0, 1, 1, 0, 0),
        incident_type="Accident / injury",
        default_message="Accident or physical injury signal detected.",
        recommended_action="Send ambulance support and request nearby safety assistance.",
        severity_hint="high",
    ),
    GestureDefinition(
        label="SECURITY",
        pattern=(0, 1, 0, 0, 0),
        incident_type="Security threat",
        default_message="Security or police assistance signal detected.",
        recommended_action="Escalate to security responders and verify the scene urgently.",
        severity_hint="high",
    ),
    GestureDefinition(
        label="SAFE",
        pattern=(1, 0, 0, 0, 0),
        incident_type="Safe / resolved",
        default_message="User indicates the situation is under control.",
        recommended_action="Keep monitoring and close the alert only after confirmation.",
        severity_hint="low",
    ),
)

UNKNOWN_GESTURE = GestureReading(
    label="UNKNOWN",
    confidence=0.0,
    incident_type="Monitoring",
    default_message="No stable emergency gesture detected.",
    recommended_action="Continue monitoring the user feed.",
    severity_hint="low",
    finger_states=(0, 0, 0, 0, 0),
)


class EmergencyGestureRecognizer:
    """Recognize a small emergency gesture vocabulary from one hand."""

    def _from_label(
        self,
        label: str,
        confidence: float,
        finger_states: Sequence[int],
    ) -> GestureReading:
        for definition in EMERGENCY_GESTURES:
            if definition.label == label:
                return GestureReading(
                    label=definition.label,
                    confidence=_clamp(confidence),
                    incident_type=definition.incident_type,
                    default_message=definition.default_message,
                    recommended_action=definition.recommended_action,
                    severity_hint=definition.severity_hint,
                    finger_states=tuple(int(value) for value in finger_states),
                )
        return UNKNOWN_GESTURE

    def recognize_finger_states(
        self,
        finger_states: Sequence[int],
        margins: Optional[Sequence[float]] = None,
    ) -> GestureReading:
        states = tuple(int(v) for v in finger_states)
        margin_score = mean(margins) if margins else 0.0

        best_definition: Optional[GestureDefinition] = None
        best_matches = -1
        for definition in EMERGENCY_GESTURES:
            matches = sum(int(a == b) for a, b in zip(states, definition.pattern))
            if matches > best_matches:
                best_matches = matches
                best_definition = definition

        if best_definition is None or best_matches < 4:
            return GestureReading(
                label="UNKNOWN",
                confidence=_clamp(0.2 + margin_score * 0.4),
                incident_type="Monitoring",
                default_message="No stable emergency gesture detected.",
                recommended_action="Continue monitoring the user feed.",
                severity_hint="low",
                finger_states=states,
            )

        confidence = _clamp(0.55 + (best_matches / 5.0) * 0.25 + margin_score * 0.35)
        return GestureReading(
            label=best_definition.label,
            confidence=confidence,
            incident_type=best_definition.incident_type,
            default_message=best_definition.default_message,
            recommended_action=best_definition.recommended_action,
            severity_hint=best_definition.severity_hint,
            finger_states=states,
        )

    def detect(self, hand_landmarks: Any, handedness_label: Optional[str]) -> GestureReading:
        """Estimate the active gesture for one detected hand."""
        landmarks = hand_landmarks.landmark
        wrist = landmarks[0]
        palm_center = _average_point([landmarks[idx] for idx in (0, 5, 9, 13, 17)])
        palm_span = max(
            _distance(landmarks[5], landmarks[17]),
            _distance(landmarks[0], landmarks[9]),
            1e-6,
        )

        finger_states: list[int] = []
        margins: list[float] = []

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        thumb_reach = (_distance(wrist, thumb_tip) - _distance(wrist, thumb_ip)) / palm_span
        thumb_spread = (_distance(thumb_tip, index_mcp) - _distance(thumb_ip, index_mcp)) / palm_span
        thumb_vertical = float(thumb_mcp.y) - float(thumb_tip.y)
        thumb_tip_to_center = _distance(thumb_tip, palm_center) / palm_span
        thumb_ip_to_center = _distance(thumb_ip, palm_center) / palm_span
        thumb_margin = max(thumb_reach, thumb_spread, thumb_vertical)
        thumb_direction_ok = thumb_vertical > 0.12 or thumb_spread > 0.10 or thumb_reach > 0.18
        thumb_extended = (
            thumb_direction_ok
            and thumb_tip_to_center > thumb_ip_to_center + 0.10
            and thumb_margin > 0.12
        )
        finger_states.append(int(thumb_extended))
        margins.append(thumb_margin)

        folded_tip_distances: list[float] = []
        extended_finger_count = 0
        for tip_idx, pip_idx, mcp_idx in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            reach_gain = (_distance(wrist, tip) - _distance(wrist, pip)) / palm_span
            vertical_gain = (float(pip.y) - float(tip.y))
            tip_to_center = _distance(tip, palm_center) / palm_span
            pip_to_center = _distance(pip, palm_center) / palm_span
            above_knuckle = float(tip.y) < float(mcp.y) + 0.03
            finger_extended = (
                reach_gain > 0.18
                and vertical_gain > 0.015
                and tip_to_center > pip_to_center + 0.08
                and above_knuckle
            )
            finger_states.append(int(finger_extended))
            margins.append(max(reach_gain, vertical_gain))
            folded_tip_distances.append(tip_to_center)
            extended_finger_count += int(finger_extended)

        non_thumb_folded = all(state == 0 for state in finger_states[1:])
        compact_fist = (
            max(folded_tip_distances, default=2.0) < 1.22
            and mean(folded_tip_distances) < 1.04
            and extended_finger_count == 0
        )
        clear_thumbs_up = (
            thumb_extended
            and non_thumb_folded
            and thumb_vertical > 0.14
            and mean(folded_tip_distances) < 1.10
        )

        if clear_thumbs_up:
            return self._from_label("SAFE", 0.9 + min(thumb_vertical, 0.08), finger_states)
        if compact_fist and thumb_tip_to_center < 1.35:
            return self._from_label("MEDICAL", 0.88 + min(0.06, thumb_ip_to_center * 0.02), finger_states)

        return self.recognize_finger_states(finger_states=finger_states, margins=margins)


class FaceDistressAnalyzer:
    """Score silent distress cues from MediaPipe face landmarks."""

    def analyze(self, face_landmarks: Any) -> DistressReading:
        points = face_landmarks.landmark

        mouth_ratio = _distance(points[13], points[14]) / max(_distance(points[78], points[308]), 1e-6)
        left_eye_ratio = _distance(points[159], points[145]) / max(_distance(points[33], points[133]), 1e-6)
        right_eye_ratio = _distance(points[386], points[374]) / max(_distance(points[362], points[263]), 1e-6)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
        left_brow_ratio = _distance(points[105], points[159]) / max(_distance(points[33], points[133]), 1e-6)
        right_brow_ratio = _distance(points[334], points[386]) / max(
            _distance(points[362], points[263]), 1e-6
        )
        brow_ratio = (left_brow_ratio + right_brow_ratio) / 2.0

        mouth_score = _scale(mouth_ratio, 0.04, 0.16)
        eye_score = _scale(eye_ratio, 0.16, 0.34)
        brow_score = _scale(brow_ratio, 0.16, 0.32)

        distress_score = _clamp(mouth_score * 0.45 + eye_score * 0.35 + brow_score * 0.20)
        if distress_score >= 0.78:
            label = "CRITICAL DISTRESS"
        elif distress_score >= 0.58:
            label = "HIGH DISTRESS"
        elif distress_score >= 0.35:
            label = "CONCERN"
        else:
            label = "CALM"

        return DistressReading(
            label=label,
            score=distress_score,
            mouth_ratio=mouth_ratio,
            eye_ratio=eye_ratio,
            brow_ratio=brow_ratio,
        )


class VERSAlertEngine:
    """Fuse gesture + distress signals into a stable live incident state."""

    def __init__(
        self,
        stable_frames: int = 6,
        history_size: int = 12,
        alert_cooldown_seconds: float = 8.0,
        log_path: Optional[Path] = None,
    ) -> None:
        self.stable_frames = max(2, int(stable_frames))
        self.gesture_history: Deque[str] = deque(maxlen=max(history_size, self.stable_frames))
        self.distress_history: Deque[float] = deque(maxlen=max(history_size, self.stable_frames))
        self.alert_cooldown_seconds = max(2.0, float(alert_cooldown_seconds))
        self.log_path = log_path
        self.last_alert_at = 0.0
        self.last_alert_signature = ""
        self.alerts: list[AlertRecord] = []

    def update(
        self,
        gesture_candidates: Sequence[GestureReading],
        distress: DistressReading,
        now: float,
    ) -> LiveStatus:
        primary_gesture = self._select_primary_gesture(gesture_candidates)
        tracked_label = primary_gesture.label if primary_gesture.confidence >= 0.6 else "UNKNOWN"
        self.gesture_history.append(tracked_label)
        self.distress_history.append(distress.score)

        stable_gesture, stable_count = self._stable_gesture()
        avg_distress = mean(self.distress_history) if self.distress_history else 0.0

        incident_type, severity, message, recommended_action, active_signal = self._build_incident(
            primary_gesture=primary_gesture,
            stable_gesture=stable_gesture,
            stable_count=stable_count,
            avg_distress=avg_distress,
            distress=distress,
        )
        status_banner = self._status_banner(severity=severity, incident_type=incident_type)

        new_alert_dict: Optional[Dict[str, Any]] = None
        signature = f"{incident_type}:{severity}:{stable_gesture}:{distress.label}"
        should_raise = (
            incident_type not in {"Monitoring", "Safe / resolved"}
            and (
                stable_count >= self.stable_frames
                or avg_distress >= 0.72
            )
            and (
                signature != self.last_alert_signature
                or (now - self.last_alert_at) >= self.alert_cooldown_seconds
            )
        )
        if should_raise:
            new_alert = self._create_alert(
                incident_type=incident_type,
                severity=severity,
                gesture=stable_gesture if stable_gesture != "UNKNOWN" else primary_gesture.label,
                gesture_confidence=primary_gesture.confidence,
                distress_label=distress.label,
                distress_score=avg_distress,
                message=message,
                recommended_action=recommended_action,
            )
            self.alerts.append(new_alert)
            self.last_alert_at = now
            self.last_alert_signature = signature
            self._append_log(new_alert)
            new_alert_dict = asdict(new_alert)

        alert_dicts = [asdict(alert) for alert in reversed(self.alerts[-6:])]
        return LiveStatus(
            gesture=stable_gesture if stable_gesture != "UNKNOWN" else primary_gesture.label,
            gesture_confidence=primary_gesture.confidence,
            distress_label=distress.label,
            distress_score=avg_distress,
            incident_type=incident_type,
            severity=severity,
            status_banner=status_banner,
            message=message,
            recommended_action=recommended_action,
            stable_frames=stable_count,
            active_signal=active_signal,
            alerts=alert_dicts,
            alert_count=len(self.alerts),
            new_alert=new_alert_dict,
            log_path=str(self.log_path) if self.log_path else None,
        )

    def _select_primary_gesture(self, gesture_candidates: Sequence[GestureReading]) -> GestureReading:
        valid = [gesture for gesture in gesture_candidates if gesture.label != "UNKNOWN"]
        if not valid:
            return UNKNOWN_GESTURE
        return max(valid, key=lambda item: item.confidence)

    def _stable_gesture(self) -> tuple[str, int]:
        if not self.gesture_history:
            return "UNKNOWN", 0
        label = self.gesture_history[-1]
        count = 0
        for item in reversed(self.gesture_history):
            if item != label:
                break
            count += 1
        return label, count

    def _build_incident(
        self,
        primary_gesture: GestureReading,
        stable_gesture: str,
        stable_count: int,
        avg_distress: float,
        distress: DistressReading,
    ) -> tuple[str, str, str, str, str]:
        active_gesture = primary_gesture if stable_count < self.stable_frames else self._lookup(stable_gesture)

        if active_gesture.label == "SAFE" and avg_distress < 0.35:
            return (
                "Safe / resolved",
                "low",
                "User indicates the situation is under control. Keep passive monitoring active.",
                "Monitor briefly, then close the event after visual confirmation.",
                "resolution signal",
            )

        if active_gesture.label != "UNKNOWN":
            severity = self._raise_severity(active_gesture.severity_hint, avg_distress)
            message = active_gesture.default_message
            if avg_distress >= 0.58:
                message += " Facial distress cues strengthen the urgency level."
            return (
                active_gesture.incident_type,
                severity,
                message,
                active_gesture.recommended_action,
                f"gesture:{active_gesture.label.lower()}",
            )

        if avg_distress >= 0.72:
            return (
                "Silent distress",
                "critical",
                "No explicit hand signal is stable, but severe facial distress is sustained.",
                "Trigger a silent welfare check and dispatch support immediately.",
                "distress-only",
            )
        if avg_distress >= 0.5:
            return (
                "Check user condition",
                "medium",
                "Possible discomfort or anxiety detected. Continue verifying the user state.",
                "Prompt the user for another gesture and keep the feed under observation.",
                "distress-monitoring",
            )
        return (
            "Monitoring",
            "low",
            "System is tracking hand gestures and facial distress indicators.",
            "No action required yet.",
            "monitoring",
        )

    def _lookup(self, label: str) -> GestureReading:
        for definition in EMERGENCY_GESTURES:
            if definition.label == label:
                return GestureReading(
                    label=definition.label,
                    confidence=1.0,
                    incident_type=definition.incident_type,
                    default_message=definition.default_message,
                    recommended_action=definition.recommended_action,
                    severity_hint=definition.severity_hint,
                    finger_states=definition.pattern,
                )
        return UNKNOWN_GESTURE

    def _raise_severity(self, base: str, distress_score: float) -> str:
        levels = ["low", "medium", "high", "critical"]
        base_index = levels.index(base)
        if distress_score >= 0.72:
            base_index = min(base_index + 1, len(levels) - 1)
        elif distress_score >= 0.5 and base_index < 2:
            base_index += 1
        return levels[base_index]

    def _status_banner(self, severity: str, incident_type: str) -> str:
        if severity == "critical":
            return f"CRITICAL ALERT: {incident_type.upper()}"
        if severity == "high":
            return f"HIGH PRIORITY: {incident_type.upper()}"
        if severity == "medium":
            return f"CHECK REQUIRED: {incident_type.upper()}"
        return f"LIVE MONITORING: {incident_type.upper()}"

    def _create_alert(
        self,
        incident_type: str,
        severity: str,
        gesture: str,
        gesture_confidence: float,
        distress_label: str,
        distress_score: float,
        message: str,
        recommended_action: str,
    ) -> AlertRecord:
        return AlertRecord(
            alert_id=uuid4().hex[:12],
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            incident_type=incident_type,
            severity=severity,
            gesture=gesture,
            gesture_confidence=_clamp(gesture_confidence),
            distress_label=distress_label,
            distress_score=_clamp(distress_score),
            message=message,
            recommended_action=recommended_action,
        )

    def _append_log(self, alert: AlertRecord) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(alert)) + "\n")


def calm_distress() -> DistressReading:
    return DistressReading(label="CALM", score=0.0, mouth_ratio=0.0, eye_ratio=0.0, brow_ratio=0.0)


def alerts_to_json(alerts: Iterable[Dict[str, Any]]) -> str:
    return json.dumps(list(alerts), indent=2)
