from pydantic import BaseModel, Field
from typing import List, Any, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    stream: bool = Field(False, description="Enable streaming (server -> client)")

class QueryResponse(BaseModel):
    success: bool
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    summary: str
    execution_time_ms: Optional[int] = None
    row_count: Optional[int] = None
    error: Optional[str] = None

class SchemaResponse(BaseModel):
    table_name: str
    db_schema: str
    columns: dict
    sample_rows: List[dict]

class HealthResponse(BaseModel):
    status: str
    database: str
    table: str

class ChatHistoryResponse(BaseModel):
    history: List[dict]
