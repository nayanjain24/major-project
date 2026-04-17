"""Tests for src/utils/data_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_utils import (
    FEATURE_DIM,
    NUM_LANDMARKS,
    csv_header,
    ensure_project_dirs,
    extract_hand_vector,
)


class TestConstants:
    def test_num_landmarks(self):
        assert NUM_LANDMARKS == 21

    def test_feature_dim(self):
        assert FEATURE_DIM == 63


class TestCsvHeader:
    def test_length(self):
        header = csv_header()
        assert len(header) == 64  # label + 63 features

    def test_first_column_is_label(self):
        header = csv_header()
        assert header[0] == "label"

    def test_feature_columns_format(self):
        header = csv_header()
        for i in range(63):
            assert header[i + 1] == f"f_{i}"


class TestExtractHandVector:
    def test_returns_none_for_no_landmarks(self):
        class FakeResults:
            multi_hand_landmarks = None

        result = extract_hand_vector(FakeResults())
        assert result is None

    def test_returns_none_for_empty_list(self):
        class FakeResults:
            multi_hand_landmarks = []

        result = extract_hand_vector(FakeResults())
        assert result is None


class TestEnsureProjectDirs:
    def test_creates_directories(self, tmp_path, monkeypatch):
        """Verify the function creates dirs without error (we just ensure it runs)."""
        # We don't want to actually create dirs in the project during tests,
        # so we just verify the function is callable and doesn't crash
        # when directories already exist.
        ensure_project_dirs()  # Should not raise
