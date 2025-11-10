import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi

from src.core.config import settings
from src.core.logging import get_logger
from src.db.connection import init_db, close_db, AsyncSessionLocal
from src.api import endpoints
from src.api.websocket_ws import router as websocket_router
from src.db.schema_inspect import SchemaInspector  

logger = get_logger(__name__)

# App Lifecycle Management

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize & close database connections."""
    logger.info("Starting FastAPI app...")
    await init_db()

    #  Preload and cache schema once for all LLM calls
    try:
        async with AsyncSessionLocal() as session:
            inspector = SchemaInspector(session)
            await inspector.get_schema_description()
            logger.info(" Database schema preloaded and cached for LLM context.")
    except Exception as e:
        logger.error(f" Failed to preload schema: {e}")

    yield
    await close_db()
    logger.info(" Shutting down FastAPI app...")


#  FastAPI Application Setup

app = FastAPI(
    title="Diamond Chat — WebSocket AI Chat",
    version="1.0.0",
    description="""💎 Diamond Chat API — FastAPI backend for NL → SQL chatbot with real-time streaming.""",
    lifespan=lifespan,
)


#  CORS Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🧩 Session Middleware

class SessionMiddleware(BaseHTTPMiddleware):
    """Generate session_id per page refresh."""

    async def dispatch(self, request: Request, call_next):
        session_id = request.cookies.get("session_id") or str(uuid.uuid4())
        request.state.session_id = session_id
        response = await call_next(request)
        response.set_cookie(key="session_id", value=session_id, httponly=False)
        return response

app.add_middleware(SessionMiddleware)


#  Routers (REST + WebSocket)

app.include_router(websocket_router)


#  Custom OpenAPI Docs

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://cdn-icons-png.flaticon.com/512/5234/5234315.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


#  Root Endpoint

@app.get("/")
async def root(request: Request):
    """Root endpoint showing session info."""
    return {
        "success": True,
        "message": "Diamond Chat API is running 🚀",
        "session_id": request.state.session_id,
        "websocket": "/ws/query",
    }
