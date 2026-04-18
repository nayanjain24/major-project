from pydantic import BaseModel


class DatasetCreate(BaseModel):
    name: str


class DatasetResponse(BaseModel):
    id: int
    name: str
    filename: str
    rows: int
    cols: int


class DatasetSummary(BaseModel):
    id: int
    name: str
    rows: int
    cols: int
