"""Research router — POST /research endpoint."""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.research_service import run_research

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    topic: str


@router.post("")
async def research_endpoint(req: ResearchRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="topic must not be empty")
    try:
        result = run_research(req.topic)
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Research pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
