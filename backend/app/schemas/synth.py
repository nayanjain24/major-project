from typing import Dict, List
from pydantic import BaseModel


class SynthesisRequest(BaseModel):
    dataset_id: int
    samples: int


class SynthesisResponse(BaseModel):
    dataset_id: int
    samples: int
    columns: List[str]
    metrics: Dict[str, float]
