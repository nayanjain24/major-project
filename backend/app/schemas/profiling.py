from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_pct: float
    unique: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class DatasetProfile(BaseModel):
    rows: int
    cols: int
    columns: List[ColumnProfile]
    missing_by_column: Dict[str, float]
    correlation_matrix: List[List[float]]
    correlation_labels: List[str]
    categorical_cardinality: Dict[str, int]
    metadata: Dict[str, Any]
