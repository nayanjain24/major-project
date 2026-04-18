from typing import Dict, List, Optional
from pydantic import BaseModel


class ImputeRequest(BaseModel):
    dataset_id: int
    target_columns: Optional[List[str]] = None


class ImputeResponse(BaseModel):
    dataset_id: int
    imputed_rows: int
    columns: List[str]
    metrics: Dict[str, float]
