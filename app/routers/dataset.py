"""Dataset router — POST /analyze-dataset endpoint."""
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.services.ds_service import run_ds

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze-dataset", tags=["dataset"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("")
async def dataset_endpoint(
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    if not target_column.strip():
        raise HTTPException(status_code=400, detail="target_column must not be empty")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    try:
        result = run_ds(contents, file.filename, target_column.strip())
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"DS pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
