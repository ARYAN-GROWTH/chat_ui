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
    logger.debug(f" Verified session in DB: {session_id}")



@router.websocket("/query")
async def websocket_query(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    """
    Handles Diamond Chat real-time communication via WebSocket.
    Steps:
      1. Accept connection and parse handshake
      2. Create or resume session
      3. Stream query processing via QueryService
      4. Stream structured HTML results + summary
    """
    await websocket.accept()
    logger.info(" WebSocket connection accepted")

    try:

        init_msg = await websocket.receive_text()
        try:
            init_data = json.loads(init_msg)
        except json.JSONDecodeError:
            await websocket.send_json({
                "success": False,
                "message": "Invalid JSON in initial handshake.",
                "data": {},
                "error": True,
                "pagination": None
            })
            await websocket.close()
            return


        session_id = init_data.get("session_id") or str(uuid.uuid4())
        await ensure_session_in_db(db, session_id)

        event_type = "session_resumed" if "session_id" in init_data else "session_created"
        message = " Session resumed" if "session_id" in init_data else " New session created"
        await websocket.send_json({
            "event": event_type,
            "success": True,
            "session_id": session_id,
            "message": message,
            "data": {},
            "error": None,
            "pagination": None
        })
        logger.info(f"[WS] {message}: {session_id}")


        service = QueryService(db, session_id=session_id)

        while True:
            try:
                raw_message = await websocket.receive_text()
                try:
                    payload = json.loads(raw_message)
                    user_query = payload.get("query", "").strip()
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "event": "error",
                        "success": False,
                        "message": "Invalid JSON payload in query.",
                        "data": {},
                        "error": True,
                        "pagination": None
                    })
                    continue

                if not user_query:
                    await websocket.send_json({
                        "event": "error",
                        "success": False,
                        "message": "Query cannot be empty.",
                        "data": {},
                        "error": True,
                        "pagination": None
                    })
                    continue

                message_id = str(uuid.uuid4())
                logger.info(f"[WS]  session={session_id}, message={message_id}, query='{user_query}'")

                # Notify client that processing has started
                await websocket.send_json({
                    "event": "status",
                    "success": True,
                    "message": " Processing query...",
                    "session_id": session_id,
                    "message_id": message_id,
                    "data": {},
                    "error": None,
                    "pagination": None
                })


                async for event in service.stream_process(user_query):
                    # Attach identifiers
                    event["session_id"] = session_id
                    event["message_id"] = message_id

                    # Ensure consistent JSON envelope
                    await websocket.send_json({
                        "success": True if event.get("event") != "error" else False,
                        "message": event.get("message", ""),
                        "data": event.get("data", {}),
                        "event": event.get("event"),
                        "session_id": session_id,
                        "message_id": message_id,
                        "error": True if event.get("event") == "error" else None,
                        "pagination": None
                    })

                await websocket.send_json({
                    "event": "complete",
                    "success": True,
                    "message": " Query execution completed.",
                    "session_id": session_id,
                    "message_id": message_id,
                    "data": {},
                    "error": None,
                    "pagination": None
                })

                await ensure_session_in_db(db, session_id)


            except WebSocketDisconnect:
                logger.info(f"[WS] 🔌 Disconnected — session={session_id}")
                break

            except Exception as e:
                logger.error(f"[WS]  Query processing error: {e}")
                await websocket.send_json({
                    "event": "error",
                    "success": False,
                    "message": str(e),
                    "session_id": session_id,
                    "data": {},
                    "error": True,
                    "pagination": None
                })
                await ensure_session_in_db(db, session_id)

    except WebSocketDisconnect:
        logger.info(" WebSocket disconnected before initialization.")
    except Exception as e:
        logger.error(f" Fatal WebSocket error: {e}")
        try:
            await websocket.send_json({
                "event": "error",
                "success": False,
                "message": str(e),
                "session_id": "unknown",
                "data": {},
                "error": True,
                "pagination": None
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(" WebSocket connection closed cleanly")
