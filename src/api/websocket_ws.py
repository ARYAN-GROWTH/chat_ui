from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.db.connection import get_db
from src.services.query_service import QueryService
from src.core.logging import get_logger
from src.core.config import settings
import json
import uuid

router = APIRouter(prefix="/ws", tags=["WebSocket"])
logger = get_logger(__name__)


#  Helper: Ensure session exists in DB

async def ensure_session_in_db(db: AsyncSession, session_id: str):
    """
    Inserts or updates a session record in the sessions table.
    Updates 'last_activity' timestamp for active sessions.
    """
    await db.execute(
        text(f"""
            INSERT INTO {settings.SCHEMA}.sessions (session_id)
            VALUES (:sid)
            ON CONFLICT (session_id)
            DO UPDATE SET last_activity = CURRENT_TIMESTAMP
        """),
        {"sid": session_id},
    )
    await db.commit()
    logger.debug(f"Verified session in DB: {session_id}")



# Helper: Standard WebSocket Response Formatter

def format_ws_response(event: dict, session_id: str, message_id: str) -> dict:
    """
    Formats all WebSocket responses into a unified structure
    consistent with the standard APIResponse format.
    """
    base = {
        "success": True,
        "event": event.get("event"),  #  moved to top-level for frontend compatibility
        "message": None,
        "data": {},
        "error": None,
        "pagination": None,
        "session_id": session_id,
        "message_id": message_id,
    }

    # Copy message if available
    if "message" in event:
        base["message"] = event["message"]

    # Merge any 'data' dicts directly
    if "data" in event and isinstance(event["data"], dict):
        base["data"].update(event["data"])

    # Handle error events gracefully
    if event.get("event") == "error":
        base["success"] = False
        base["error"] = event.get("message", "Unknown error")
        if not base["message"]:
            base["message"] = " An error occurred while processing your query."

    # Copy other dynamic keys like 'chunk', 'sql', etc.
    for k, v in event.items():
        if k not in ["event", "message", "data"]:
            base["data"][k] = v

    # Default message if missing
    if not base["message"]:
        base["message"] = f"Event: {event.get('event', 'unknown')}"

    return base



#  Main WebSocket Endpoint — Real-time Query Processing

@router.websocket("/query")
async def websocket_query(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """
    Handles Diamond Chat real-time communication via WebSocket.
    Steps:
      1 Accept connection and parse handshake
      2 Create or resume session
      3 Stream query processing via QueryService
      4 Send unified responses (success, message, data, error, pagination)
    """
    await websocket.accept()
    logger.info(" WebSocket connection accepted")

    try:
        # Step  — Initial handshake
        init_msg = await websocket.receive_text()
        try:
            init_data = json.loads(init_msg)
        except json.JSONDecodeError:
            await websocket.send_json({
                "success": False,
                "event": "error",
                "message": "Invalid JSON in initial handshake.",
                "data": {},
                "error": True,
                "pagination": None
            })
            await websocket.close()
            return

        # Step 2️ — Create or resume session
        session_id = init_data.get("session_id") or str(uuid.uuid4())
        await ensure_session_in_db(db, session_id)

        event_type = "session_resumed" if "session_id" in init_data else "session_created"
        message = "Session resumed" if "session_id" in init_data else "New session created"

        await websocket.send_json(format_ws_response({
            "event": event_type,
            "message": message,
        }, session_id, "init"))

        logger.info(f"[WS] {message}: {session_id}")

        # Step 3️ — Initialize QueryService
        service = QueryService(db, session_id=session_id)

        # Step 4️ — Main message loop
        while True:
            try:
                raw_message = await websocket.receive_text()
                try:
                    payload = json.loads(raw_message)
                    user_query = payload.get("query", "").strip()
                except json.JSONDecodeError:
                    await websocket.send_json(format_ws_response({
                        "event": "error",
                        "message": "Invalid JSON payload in query."
                    }, session_id, "parse-error"))
                    continue

                if not user_query:
                    await websocket.send_json(format_ws_response({
                        "event": "error",
                        "message": "Query cannot be empty."
                    }, session_id, "empty-query"))
                    continue

                message_id = str(uuid.uuid4())
                logger.info(f"[WS] session={session_id}, message={message_id}, query='{user_query}'")

                # Notify client query processing started
                await websocket.send_json(format_ws_response({
                    "event": "status",
                    "message": "Processing query..."
                }, session_id, message_id))

                # Stream results from QueryService
                async for event in service.stream_process(user_query):
                    formatted = format_ws_response(event, session_id, message_id)
                    await websocket.send_json(formatted)
                    logger.debug(f"[WS -> Client] Sent event: {formatted.get('event')}")

                # Send final completion event
                await websocket.send_json(format_ws_response({
                    "event": "complete",
                    "message": "Query execution completed."
                }, session_id, message_id))

                await ensure_session_in_db(db, session_id)

            except WebSocketDisconnect:
                logger.info(f"[WS] 🔌 Disconnected — session={session_id}")
                break

            except Exception as e:
                logger.error(f"[WS] Query processing error: {e}")
                await websocket.send_json(format_ws_response({
                    "event": "error",
                    "message": str(e)
                }, session_id, "query-error"))
                await ensure_session_in_db(db, session_id)

    except WebSocketDisconnect:
        logger.info(" WebSocket disconnected before initialization.")
    except Exception as e:
        logger.error(f" Fatal WebSocket error: {e}")
        try:
            await websocket.send_json(format_ws_response({
                "event": "error",
                "message": str(e)
            }, "unknown", "fatal"))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(" WebSocket connection closed cleanly")
