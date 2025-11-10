from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pathlib import Path
import uuid

from src.db.connection import get_db
from src.db.schema_inspect import SchemaInspector
from src.services.query_service import QueryService
from src.api.models import SchemaResponse, HealthResponse, ChatHistoryResponse
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Diamond Chat API"])


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(db: AsyncSession = Depends(get_db)):
    """
    Returns the schema structure and sample data for the configured table.
    """
    inspector = SchemaInspector(db)
    columns = await inspector.get_table_schema()
    samples = await inspector.get_sample_rows(limit=5)

    return SchemaResponse(
        table_name=settings.TABLE_NAME,
        db_schema=settings.SCHEMA,
        columns=columns,
        sample_rows=samples
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Checks connectivity with the database and verifies schema/table availability.
    """
    try:
        await db.execute(text("SELECT 1"))
        db_name = settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "configured"
        return HealthResponse(
            status="ok",
            database=db_name,
            table=f"{settings.SCHEMA}.{settings.TABLE_NAME}"
        )
    except Exception as e:
        logger.error(f" Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieve recent chat history for a specific session.
    """
    svc = QueryService(db, session_id=session_id)
    history = await svc.get_chat_history(limit=50)  # type: ignore
    return ChatHistoryResponse(history=history)

@router.delete("/history/{session_id}")
async def clear_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Clear all chat messages related to a session.
    """
    svc = QueryService(db, session_id=session_id)
    await svc.clear_history()  # type: ignore
    return {
        "success": True,
        "message": f"Chat history cleared for session {session_id}",
        "data": {},
        "error": None,
        "pagination": None
    }


@router.get("/downloads/{filename}")
async def download_excel(filename: str):
    """
    Serves .xlsx files generated during query execution.
    Example: GET /api/v1/downloads/export_<session>_<timestamp>.xlsx
    """
    exports_dir = Path("exports")
    file_path = exports_dir / filename

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {
            "success": False,
            "message": "File not found.",
            "data": {},
            "error": True,
            "pagination": None
        }

    logger.info(f" Sending Excel file: {file_path.name}")
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
