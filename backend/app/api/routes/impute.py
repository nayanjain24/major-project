from fastapi import APIRouter, HTTPException

from app.schemas.export import ExportResponse
from app.schemas.impute import ImputeRequest, ImputeResponse
from app.services.datasets import DatasetRegistry
from app.services.exports import ExportService
from app.services.imputation import ImputationService

router = APIRouter()


@router.post("/impute", response_model=ImputeResponse)
def impute_dataset(payload: ImputeRequest) -> ImputeResponse:
    registry = DatasetRegistry()
    service = ImputationService()

    try:
        df = registry.load_dataframe(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    imputed, metrics = service.impute(df, payload.target_columns)
    return ImputeResponse(
        dataset_id=payload.dataset_id,
        imputed_rows=len(imputed),
        columns=imputed.columns.tolist(),
        metrics=metrics,
    )


@router.post("/impute/{dataset_id}/export", response_model=ExportResponse)
def export_imputed(dataset_id: int) -> ExportResponse:
    registry = DatasetRegistry()
    service = ImputationService()
    exporter = ExportService()

    try:
        df = registry.load_dataframe(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    imputed, _ = service.impute(df, None)
    output = exporter.export_csv(imputed, f"imputed_{dataset_id}.csv")
    return ExportResponse(dataset_id=dataset_id, file_path=str(output))
