from typing import Dict
from pydantic import BaseModel


class MetricsResponse(BaseModel):
    dataset_id: int
    metrics: Dict[str, float]
