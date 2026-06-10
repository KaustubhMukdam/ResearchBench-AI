"""FastAPI app entry point."""
import os
import logging
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGSMITH_TRACING", "true")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.research import router as research_router
from app.routers.dataset import router as dataset_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchBench AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)
app.include_router(dataset_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
