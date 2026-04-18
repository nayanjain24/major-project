from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.core.config import settings
from app.utils.file import ensure_dir


class DatasetRegistry:
    def __init__(self) -> None:
        self.base_dir = ensure_dir(settings.storage_dir)
        self.registry_path = self.base_dir / "registry.json"
        if not self.registry_path.exists():
            self.registry_path.write_text("[]")

    def _read(self) -> List[Dict]:
        return json.loads(self.registry_path.read_text())

    def _write(self, data: List[Dict]) -> None:
        self.registry_path.write_text(json.dumps(data, indent=2))

    def add(self, name: str, filename: str, rows: int, cols: int) -> Dict:
        data = self._read()
        next_id = max([item["id"] for item in data], default=0) + 1
        record = {
            "id": next_id,
            "name": name,
            "filename": filename,
            "rows": rows,
            "cols": cols,
        }
        data.append(record)
        self._write(data)
        return record

    def list(self) -> List[Dict]:
        return self._read()

    def get(self, dataset_id: int) -> Dict:
        data = self._read()
        for item in data:
            if item["id"] == dataset_id:
                return item
        raise KeyError(f"Dataset {dataset_id} not found")

    def load_dataframe(self, dataset_id: int) -> pd.DataFrame:
        record = self.get(dataset_id)
        return self.load_dataframe_from_path(Path(record["filename"]))

    def load_dataframe_from_path(self, file_path: Path) -> pd.DataFrame:
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        return pd.read_csv(file_path)
