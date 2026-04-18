from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.schemas.dataset import DatasetResponse, DatasetSummary
from app.services.datasets import DatasetRegistry
from app.services.storage import LocalStorage

router = APIRouter()


@router.post("/datasets", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetResponse:
    storage = LocalStorage()
    registry = DatasetRegistry()

    suffix = Path(file.filename).suffix
    safe_name = file.filename.replace(" ", "_")
    stored = storage.save(safe_name, file.file)

    df = registry.load_dataframe_from_path(stored)
    record = registry.add(name=file.filename, filename=str(stored), rows=len(df), cols=len(df.columns))
    return DatasetResponse(**record)


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets() -> list[DatasetSummary]:
    registry = DatasetRegistry()
    return [DatasetSummary(**item) for item in registry.list()]
