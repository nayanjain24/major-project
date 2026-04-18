from fastapi import APIRouter, HTTPException

from app.schemas.export import ExportResponse
from app.schemas.synth import SynthesisRequest, SynthesisResponse
from app.services.datasets import DatasetRegistry
from app.services.exports import ExportService
from app.services.synthesis import SynthesisService

router = APIRouter()


@router.post("/synthesize", response_model=SynthesisResponse)
def synthesize_dataset(payload: SynthesisRequest) -> SynthesisResponse:
    registry = DatasetRegistry()
    service = SynthesisService()

    try:
        df = registry.load_dataframe(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    generated, metrics = service.generate(df, payload.samples)
    return SynthesisResponse(
        dataset_id=payload.dataset_id,
        samples=len(generated),
        columns=generated.columns.tolist(),
        metrics=metrics,
    )


@router.post("/synthesize/{dataset_id}/export", response_model=ExportResponse)
def export_synthetic(dataset_id: int, samples: int = 1000) -> ExportResponse:
    registry = DatasetRegistry()
    service = SynthesisService()
    exporter = ExportService()

    try:
        df = registry.load_dataframe(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    generated, _ = service.generate(df, samples)
    output = exporter.export_csv(generated, f"synthetic_{dataset_id}.csv")
    return ExportResponse(dataset_id=dataset_id, file_path=str(output))
