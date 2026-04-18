from fastapi import APIRouter, HTTPException

from app.schemas.prompt import PromptOptimizationRequest, PromptOptimizationResponse
from app.services.prompt_optimizer import (
    PromptOptimizerConfigError,
    PromptOptimizerService,
    PromptOptimizerUpstreamError,
)

router = APIRouter()


@router.post("/prompts/optimize", response_model=PromptOptimizationResponse)
def optimize_prompt(payload: PromptOptimizationRequest) -> PromptOptimizationResponse:
    service = PromptOptimizerService()
    try:
        optimized = service.optimize(payload)
    except PromptOptimizerConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PromptOptimizerUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PromptOptimizationResponse(**optimized.__dict__)
