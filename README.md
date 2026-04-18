# DataFusion AI

DataFusion AI is a production-ready concept build for diffusion-based tabular data enhancement (imputation + synthetic data generation), with fairness-aware diagnostics and a clean workflow UI.

## Prompt Optimizer Quickstart

The frontend currently ships a PromptPilot-style workflow that generates optimized prompts for everyday questions using the backend AI endpoint.

Backend setup:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set DATAFUSION_OPENAI_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Frontend setup:
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## 1. System Architecture (Textual Diagram)

**Client (React + Tailwind)**
- Uploads datasets
- Shows profiling + metrics + workflow steps
- Triggers diffusion jobs

**API Gateway (FastAPI)**
- `POST /api/datasets` ingestion
- `GET /api/datasets/{id}/profile` profiling
- `POST /api/impute` diffusion imputation
- `POST /api/synthesize` diffusion synthesis
- `GET /api/datasets/{id}/metrics` evaluation
- `POST /api/prompts/optimize` AI prompt optimization
- Export endpoints for imputed/synthetic datasets

**ML Pipeline (PyTorch)**
- Preprocessing (scaler + one-hot)
- TabDDPM diffusion model
- Mask-aware imputation (scaffolded)
- Synthetic sampling

**Storage**
- Local file storage (CSV/XLSX) for this build
- PostgreSQL schema included for production

**Security**
- AES-256 encryption helper (Fernet)
- OAuth2/Firebase hooks ready for extension

## 2. Frontend Component Structure

- `src/App.tsx`: Hero + overall layout
- `src/components/Workflow.tsx`: Step-by-step workflow UI
- `src/components/StepCard.tsx`: Styled step block
- `src/components/MetricPill.tsx`: Metric badges
- `src/lib/api.ts`: Backend API client

## 3. Backend API Design

**Dataset**
- `POST /api/datasets` (multipart) -> upload
- `GET /api/datasets` -> list datasets
- `GET /api/datasets/{id}/profile` -> profiling stats

**Diffusion**
- `POST /api/impute` -> impute missing values
- `POST /api/synthesize` -> generate synthetic rows

**Exports**
- `POST /api/impute/{id}/export`
- `POST /api/synthesize/{id}/export?samples=1000`

**Metrics**
- `GET /api/datasets/{id}/metrics` -> KS-based distribution similarity

**Prompting**
- `POST /api/prompts/optimize` -> generate optimized prompt + quality feedback

## 4. Diffusion Model Pseudocode

```text
for each epoch:
  for each batch x:
    t ~ Uniform(0, T)
    noise ~ Normal(0, 1)
    x_t = sqrt(alpha_bar[t]) * x + sqrt(1 - alpha_bar[t]) * noise
    pred_noise = denoiser(x_t, t)
    loss = MSE(pred_noise, noise)
    backprop + update

Sampling:
  x_T ~ Normal(0, 1)
  for t = T-1 ... 0:
    x_{t-1} = p_sample(x_t, t)
```

## 5. Database Schema

See `backend/app/db/schema.sql` for:
- `datasets`
- `jobs`
- `metric_snapshots`

## 6. Deployment (Docker)

```bash
cd backend
docker compose up --build
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

Prompt optimization setup:
```bash
export DATAFUSION_OPENAI_API_KEY=your_openai_api_key
export DATAFUSION_OPENAI_MODEL=gpt-4o-mini  # optional
```

## 7. Sample UI Flow

1. Upload dataset
2. Profile for missingness + correlations
3. Run diffusion imputation
4. Generate synthetic data
5. Evaluate distribution similarity
6. Export datasets

## 8. Notes

- Diffusion modules are scaffolded for speed; increase epochs/steps for better results.
- Mask-aware denoising and fairness-aware sampling hooks are ready for extension.
