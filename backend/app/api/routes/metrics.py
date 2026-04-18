from fastapi import APIRouter, HTTPException

from app.schemas.metrics import MetricsResponse
from app.services.datasets import DatasetRegistry
from app.services.metrics import MetricsService
from app.services.synthesis import SynthesisService

router = APIRouter()


@router.get("/datasets/{dataset_id}/metrics", response_model=MetricsResponse)
def dataset_metrics(dataset_id: int) -> MetricsResponse:
    registry = DatasetRegistry()
    metrics_service = MetricsService()
    synth_service = SynthesisService()

    try:
        df = registry.load_dataframe(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    synthetic, _ = synth_service.generate(df, samples=min(200, len(df)))
    metrics = metrics_service.distribution_similarity(df, synthetic)
    return MetricsResponse(dataset_id=dataset_id, metrics=metrics)
