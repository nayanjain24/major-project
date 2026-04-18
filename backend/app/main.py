from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import datasets, health, impute, metrics, profiling, prompts, synth
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(profiling.router, prefix="/api")
app.include_router(impute.router, prefix="/api")
app.include_router(synth.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(prompts.router, prefix="/api")
