from pydantic import BaseModel


class ExportResponse(BaseModel):
    dataset_id: int
    file_path: str
