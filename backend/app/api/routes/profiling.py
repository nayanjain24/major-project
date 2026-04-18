from fastapi import APIRouter, HTTPException

from app.schemas.profiling import DatasetProfile
from app.services.datasets import DatasetRegistry
from app.services.profiling import DataProfiler

router = APIRouter()


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfile)
def profile_dataset(dataset_id: int) -> DatasetProfile:
    registry = DatasetRegistry()
    profiler = DataProfiler()

    try:
        df = registry.load_dataframe(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = profiler.profile(df)
    return DatasetProfile(**result.__dict__)
