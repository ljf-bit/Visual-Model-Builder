"""Visual Model Builder FastAPI backend."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router as api_router


LOCAL_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _parse_cors_origins() -> list[str]:
    configured = os.environ.get("VMB_CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    return origins or LOCAL_CORS_ORIGINS

app = FastAPI(
    title="Visual Model Builder API",
    description="Backend API for the Visual Model Builder teaching tool",
    version="1.0.0",
)

cors_origins = _parse_cors_origins()
allow_all_origins = "*" in cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_origin_regex=os.environ.get("VMB_CORS_ORIGIN_REGEX") or None,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
