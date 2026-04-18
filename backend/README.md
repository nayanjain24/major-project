# DataFusion AI Backend

This backend provides dataset ingestion, profiling, diffusion-based imputation, and synthetic data generation for tabular data.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set DATAFUSION_OPENAI_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

## Core Endpoints

- `POST /api/datasets` (multipart upload)
- `GET /api/datasets`
- `GET /api/datasets/{dataset_id}/profile`
- `POST /api/impute`
- `POST /api/synthesize`
- `POST /api/prompts/optimize`

## Notes

- Storage uses `./storage` by default.
- Diffusion training is lightweight for demo; increase epochs/steps for higher quality.
- For prompt optimization, set `DATAFUSION_OPENAI_API_KEY` (and optionally `DATAFUSION_OPENAI_MODEL`).
