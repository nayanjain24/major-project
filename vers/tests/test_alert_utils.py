"""Tests for src/utils/alert_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.alert_utils import (
    ALERT_MAP,
    SEVERITY_ORDER,
    calculate_fused_severity,
    make_alert_payload,
    normalize_label,
)


class TestNormalizeLabel:
    def test_uppercase(self):
        assert normalize_label("sos") == "SOS"

    def test_whitespace(self):
        assert normalize_label("  emergency ") == "EMERGENCY"

    def test_empty_string(self):
        assert normalize_label("") == "NONE"

    def test_none_value(self):
        assert normalize_label(None) == "NONE"  # type: ignore[arg-type]

    def test_already_upper(self):
        assert normalize_label("MEDICAL") == "MEDICAL"


class TestAlertMap:
    def test_all_labels_have_required_keys(self):
        for label, info in ALERT_MAP.items():
            assert "message" in info, f"Missing 'message' for {label}"
            assert "severity" in info, f"Missing 'severity' for {label}"

    def test_severity_values_are_valid(self):
        for label, info in ALERT_MAP.items():
            assert info["severity"] in SEVERITY_ORDER, (
                f"Invalid severity '{info['severity']}' for {label}"
            )

    def test_expected_labels_present(self):
        for label in ("SOS", "EMERGENCY", "ACCIDENT", "MEDICAL", "SAFE", "NONE"):
            assert label in ALERT_MAP, f"Missing label '{label}' from ALERT_MAP"


class TestCalculateFusedSeverity:
    def test_high_confidence_high_distress_returns_critical(self):
        severity, score = calculate_fused_severity(0.95, 0.12, "High")
        assert severity == "Critical"
        assert 0.0 <= score <= 1.0

    def test_low_confidence_low_distress(self):
        severity, score = calculate_fused_severity(0.3, 0.01, "Low")
        assert severity in ("Low", "Medium")
        assert 0.0 <= score <= 1.0

    def test_never_drops_below_base(self):
        severity, _ = calculate_fused_severity(0.1, 0.0, "High")
        assert SEVERITY_ORDER[severity] >= SEVERITY_ORDER["High"]

    def test_score_clamped(self):
        _, score = calculate_fused_severity(1.5, -0.1, "Low")
        assert 0.0 <= score <= 1.0


class TestMakeAlertPayload:
    def test_required_fields_present(self):
        payload = make_alert_payload("SOS", 0.85, 0.07, True)
        required = {
            "AlertID", "Timestamp", "Location", "Severity", "BaseSeverity",
            "FusionScore", "MainGesture", "GestureConfidence", "DistressScore",
            "DistressFlag", "Message", "SecondaryEmotion",
        }
        assert required.issubset(set(payload.keys()))

    def test_gesture_is_normalized(self):
        payload = make_alert_payload("sos", 0.9, 0.05, False)
        assert payload["MainGesture"] == "SOS"

    def test_distress_flag_types(self):
        payload = make_alert_payload("SAFE", 0.8, 0.01, False)
        assert isinstance(payload["DistressFlag"], bool)
        assert isinstance(payload["GestureConfidence"], float)
        assert isinstance(payload["DistressScore"], float)
        assert isinstance(payload["FusionScore"], float)

    def test_unknown_label_falls_back_to_none(self):
        payload = make_alert_payload("UNKNOWN_GESTURE", 0.5, 0.0, False)
        assert payload["MainGesture"] == "UNKNOWN_GESTURE"
        assert payload["Message"] == ALERT_MAP["NONE"]["message"]
