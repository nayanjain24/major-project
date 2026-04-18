from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.storage import LocalStorage


class ExportService:
    def __init__(self) -> None:
        self.storage = LocalStorage()

    def export_csv(self, df: pd.DataFrame, filename: str) -> Path:
        content = df.to_csv(index=False).encode("utf-8")
        return self.storage.save_bytes(filename, content)
