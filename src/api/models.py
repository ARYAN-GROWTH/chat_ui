from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict

class APIResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the operation was successful")
    message: str = Field(..., description="Descriptive message about the API response")
    data: Optional[Any] = Field(None, description="Main payload data returned by the API")
    error: Optional[Any] = Field(None, description="Error details if any")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination info if applicable")


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
