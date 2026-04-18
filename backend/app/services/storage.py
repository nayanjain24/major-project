from pathlib import Path
from typing import BinaryIO

from app.core.config import settings
from app.utils.file import ensure_dir


class LocalStorage:
    def __init__(self) -> None:
        self.base_dir = ensure_dir(settings.storage_dir)

    def save(self, filename: str, content: BinaryIO) -> Path:
        target = self.base_dir / filename
        with target.open("wb") as handle:
            handle.write(content.read())
        return target

    def save_bytes(self, filename: str, content: bytes) -> Path:
        target = self.base_dir / filename
        with target.open("wb") as handle:
            handle.write(content)
        return target
