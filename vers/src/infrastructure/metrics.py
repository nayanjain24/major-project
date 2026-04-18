"""Lightweight in-process metrics collector for VERS v3.0.

Tracks alert counts, FPS statistics, and uptime without requiring
an external metrics server (Prometheus/Grafana).  Metrics are exportable
as JSON via the /api/v1/stats endpoint.
"""

from __future__ import annotations

import threading
import time
from collections import Counter


class MetricsCollector:
    """Thread-safe singleton metrics aggregator."""

    _instance: MetricsCollector | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricsCollector:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        self._start_time = time.time()
        self._alert_count = 0
        self._alerts_by_threat: Counter[str] = Counter()
        self._fps_samples: list[float] = []
        self._max_fps_samples = 100
        self._last_gesture = "NONE"
        self._last_emotion = "neutral"
        self._data_lock = threading.Lock()

    def record_alert(self, threat_level: str) -> None:
        with self._data_lock:
            self._alert_count += 1
            self._alerts_by_threat[threat_level] += 1

    def record_fps(self, fps: float) -> None:
        with self._data_lock:
            self._fps_samples.append(fps)
            if len(self._fps_samples) > self._max_fps_samples:
                self._fps_samples = self._fps_samples[-self._max_fps_samples:]

    def record_detection(self, gesture: str, emotion: str) -> None:
        with self._data_lock:
            self._last_gesture = gesture
            self._last_emotion = emotion

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def snapshot(self) -> dict:
        with self._data_lock:
            avg_fps = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 0.0
            return {
                "total_alerts": self._alert_count,
                "alerts_by_threat": dict(self._alerts_by_threat),
                "average_fps": round(avg_fps, 1),
                "uptime_seconds": round(self.uptime_seconds, 1),
                "last_gesture": self._last_gesture,
                "last_emotion": self._last_emotion,
            }


# Convenience accessor
def get_metrics() -> MetricsCollector:
    return MetricsCollector()
