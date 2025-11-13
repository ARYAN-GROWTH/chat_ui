# diamond_backend_final.py
import os
import re
import json
import time
import uuid
import math
import asyncio
import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# LangChain LLM (only LLM; no embeddings used)
from langchain_openai import ChatOpenAI

# ----------------------------
# Config / env
# ----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in environment")

DATABASE_URL_ASYNC = os.getenv(
    "DATABASE_URL_ASYNC",
    "postgresql://diamond_user:StrongPassword123!@postgres-db-16-3-r1.cimgrjr0vadx.ap-south-1.rds.amazonaws.com:5432/diamond-db-dev"
)

SCHEMA = os.getenv("SCHEMA", "public")
TABLE_NAME = os.getenv("TABLE_NAME", "dev_diamond2")

SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", "600"))
LAST_N_MEMORY = int(os.getenv("LAST_N_MEMORY", "3"))  # user chose 3
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50"))

# Where generated files are saved. This will be served at /downloads/<file>
USER_HOME = os.path.expanduser("~")
DOWNLOAD_DIR = os.path.join(USER_HOME, "Downloads", "download")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Also allow local dumps folder as fallback (not necessary but handy)
DUMPS_DIR = os.path.join(os.getcwd(), "dumps")
os.makedirs(DUMPS_DIR, exist_ok=True)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Diamond DB AI — Minimal Events (final)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve downloads so browser can GET and trigger "Save as..." to user's Downloads
app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")

# ----------------------------
# Pools / cache / memory
# ----------------------------
_pg_pool: Optional[asyncpg.pool.Pool] = None
_schema_cache: Dict[str, Any] = {"value": None, "expires_at": 0}

# session-scoped in-memory text memory
# session_memory[session_id] = [ {"q": "...", "a": "..."}, ... ]
session_memory: Dict[str, List[Dict[str, Any]]] = {}
session_lock = asyncio.Lock()

# ----------------------------
# LLM models
# ----------------------------
SQL_MODEL = ChatOpenAI(model=os.getenv("SQL_MODEL_NAME", "gpt-4.1-mini"), temperature=0)
ANSWER_MODEL = ChatOpenAI(model=os.getenv("ANSWER_MODEL_NAME", "gpt-4o-mini"), temperature=0)

# ----------------------------
# Utilities
# ----------------------------
def safe_to_text(resp: Any) -> str:
    """Turn LLM response or mixed object into a plain text string."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        return " ".join(safe_to_text(x) for x in resp)
    if isinstance(resp, dict):
        if "content" in resp:
            return str(resp["content"])
        return " ".join(str(v) for v in resp.values())
    if hasattr(resp, "content"):
        try:
            return str(resp.content)
        except Exception:
            return str(resp)
    return str(resp)

def serialize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v

def serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: serialize_value(v) for k, v in row.items()}

# ----------------------------
# DB helpers
# ----------------------------
async def get_pg_pool() -> asyncpg.pool.Pool:
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(DATABASE_URL_ASYNC, min_size=1, max_size=10)
    return _pg_pool

async def get_schema() -> List[Dict[str, Any]]:
    now = time.time()
    if _schema_cache["value"] and _schema_cache["expires_at"] > now:
        return _schema_cache["value"]
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
            """,
            SCHEMA, TABLE_NAME
        )
    cols = [dict(r) for r in rows]
    _schema_cache["value"] = cols
    _schema_cache["expires_at"] = now + SCHEMA_CACHE_TTL
    return cols

# ----------------------------
# Fallback intent detection
# ----------------------------
def detect_intent_fallback(question: str, schema_cols: List[Dict[str, Any]]) -> Optional[str]:
    q = (question or "").lower().strip()
    if not q:
        return None
    if "database" in q or "table" in q or "columns" in q or "schema" in q:
        # return quick summary query
        cols = [c["column_name"] for c in schema_cols]
        col_list = ", ".join(f'"{c}"' for c in cols[:20])  # limit large list
        return f"SELECT {col_list} FROM public.{TABLE_NAME} LIMIT 1"
    if "main category" in q or q in {"main_category", "main category"}:
        return f'SELECT DISTINCT "main_category" FROM public.{TABLE_NAME} LIMIT 200'
    if "subcategory" in q:
        return f'SELECT DISTINCT "subcategory" FROM public.{TABLE_NAME} LIMIT 200'
    # item_no pattern
    m = re.search(r'([0-9A-Za-z\-_.]+)', question)
    if "item_no" in q or "item no" in q or (m and "-" in (m.group(1) or "")):
        # Try to extract token that looks like code (keep flexible). Let LLM refine
        return None
    return None

# ----------------------------
# SQL sanitizer
# ----------------------------
SQL_PROMPT_TEMPLATE = """
You are an expert PostgreSQL assistant.
Return ONE valid PostgreSQL SELECT statement only (no explanation).

Table: {schema}.{table}
Columns: {columns}
Memory: {memory}

User question: {question}

Rules:
- Use explicit column names when possible.
- Use double quotes for columns with spaces/special chars.
- If 'date' column is referenced and in non-standard format, the application will try to handle/parse for Excel.
- Prefer safe, read-only SELECT statements.
- Return ONLY the SQL statement.
"""

def sanitize_sql(raw: Any) -> str:
    txt = safe_to_text(raw).strip()
    # remove potential prefix content=...
    txt = re.sub(r"^\s*content\s*[:=]\s*", "", txt, flags=re.I)
    # remove code fences
    txt = re.sub(r"```(?:sql)?", "", txt, flags=re.I).replace("```", "")
    txt = txt.replace("`", "")
    # unescape
    txt = txt.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    txt = txt.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    m = re.search(r"(select\b[\s\S]*?)(?:;|$)", txt, flags=re.I)
    if not m:
        raise ValueError("No SELECT found in LLM output.")
    sql = m.group(1).strip()
    sql = sql.split(";")[0].strip()
    return sql

# ----------------------------
# SQL generation (uses text history last N)
# ----------------------------
async def generate_sql(session_id: str, question: str) -> str:
    schema_cols = await get_schema()
    async with session_lock:
        mem = session_memory.get(session_id, [])  # full session memory
    # build last N pairs
    last_pairs = mem[-LAST_N_MEMORY:] if mem else []
    mem_text = " ; ".join(f"Q: {p.get('q','')} A: {p.get('a','')}" for p in last_pairs)
    fallback = detect_intent_fallback(question, schema_cols)
    if fallback:
        return fallback
    columns = ", ".join(c["column_name"] for c in schema_cols)
    prompt = SQL_PROMPT_TEMPLATE.format(
        schema=SCHEMA,
        table=TABLE_NAME,
        columns=columns,
        memory=mem_text,
        question=question
    )
    resp = await asyncio.to_thread(SQL_MODEL.invoke, prompt)
    sql_raw = safe_to_text(resp)
    sql = sanitize_sql(sql_raw)
    # Ensure read-only
    if not sql.lower().startswith("select"):
        raise ValueError("LLM did not produce a SELECT statement.")
    # Do NOT force LIMIT for item_no: user wanted all rows matching item_no
    return sql

# ----------------------------
# Exec SQL with retry
# ----------------------------
async def exec_sql(sql: str) -> List[Dict[str, Any]]:
    low = sql.lower()
    forbidden = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate "]
    if any(f in low for f in forbidden):
        raise ValueError("Only SELECT statements allowed.")
    pool = await get_pg_pool()
    try:
        async with pool.acquire() as conn:
            await conn.reset()
            await conn.execute("SET statement_timeout='10000';")
            rows = await conn.fetch(sql)
    except Exception:
        # retry once
        new_conn = await pool.acquire()
        try:
            await new_conn.reset()
            await new_conn.execute("SET statement_timeout='10000';")
            rows = await new_conn.fetch(sql)
        finally:
            await pool.release(new_conn)
    return [dict(r) for r in rows]

# ----------------------------
# Answer generation (short)
# ----------------------------
ANSWER_PROMPT = """
Summarize SQL results concisely.

Question: {question}
SQL: {sql}
Rows (sample up to 10): {rows}

- Short and polite.
- If empty result: say it politely.
- Provide exactly 2 follow-up suggestions.
Return only Markdown.
"""

async def generate_answer(question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
    prompt = ANSWER_PROMPT.format(
        question=question,
        sql=sql,
        rows=json.dumps(rows[:10], default=str)
    )
    resp = await asyncio.to_thread(ANSWER_MODEL.invoke, prompt)
    return safe_to_text(resp).strip()

# ----------------------------
# Date parsing for excel
# ----------------------------
def try_parse_date(value: Any) -> Optional[datetime.date]:
    if value is None:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.date() if isinstance(value, datetime.datetime) else value
    s = str(value).strip()
    if not s:
        return None
    formats = [
        "%d-%b-%y", "%d-%b-%Y",
        "%d-%B-%y", "%d-%B-%Y",
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
        "%d %b %y", "%d %b %Y", "%d %B %Y", "%d %B %y"
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            pass
    # remove st/nd/rd/th
    s2 = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.I)
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s2, fmt)
            return dt.date()
        except Exception:
            pass
    return None

# ----------------------------
# Excel writer with styling & preview
# ----------------------------
def write_styled_excel_and_preview(rows: List[Dict[str, Any]], filename: str, preview_count: int = 8) -> Dict[str, Any]:
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    wb = Workbook()
    sheet: Worksheet = wb.active  # type: ignore[assignment]
    sheet.title = "Results"
    if not rows:
        wb.save(filepath)
        return {"filename": filename, "preview_rows": []}
    headers = list(rows[0].keys())
    # header styling
    for col_idx, header in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor="1E4F9C")
        cell.alignment = Alignment(horizontal="center", vertical="center")
    # Write data rows
    for r_idx, row in enumerate(rows, start=2):
        for c_idx, header in enumerate(headers, start=1):
            raw_val = row.get(header, "")
            parsed_date = try_parse_date(raw_val)
            if parsed_date:
                sheet.cell(row=r_idx, column=c_idx, value=parsed_date)
            else:
                sheet.cell(row=r_idx, column=c_idx, value=raw_val)
    # Auto column width
    for col_idx in range(1, len(headers) + 1):
        max_len = 0
        for row_tuple in sheet.iter_rows(min_row=1, min_col=col_idx, max_col=col_idx):
            cell = row_tuple[0]
            try:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            except Exception:
                pass
        sheet.column_dimensions[get_column_letter(col_idx)].width = max(max_len + 4, 10)
    wb.save(filepath)
    preview_rows = [serialize_row(r) for r in rows[:preview_count]]
    return {"filename": filename, "preview_rows": preview_rows}

# ----------------------------
# WebSocket endpoint (minimal events)
# ----------------------------
GREETINGS = {"hello", "hi", "hey", "hola", "namaste", "hlo", "how are you", "how r you"}

@app.websocket("/ws/query")
async def ws_query(ws: WebSocket):
    # Create unique session for connection
    session_id = uuid.uuid4().hex
    async with session_lock:
        session_memory[session_id] = []
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                await ws.send_json({"type": "status", "message": "Invalid JSON payload"})
                continue

            question = safe_to_text(payload.get("question") or "").strip()
            if not question:
                await ws.send_json({"type": "status", "message": "Please send a non-empty question."})
                continue

            # Greeting quick reply (no memory, no SQL)
            if question.lower() in GREETINGS:
                await ws.send_json({"type": "status", "message": "Quick reply..."})
                await ws.send_json({"type": "answer", "markdown": "👋 Hello! How can I help you explore the Diamond database?"})
                continue

            # Status
            await ws.send_json({"type": "status", "message": "Working on your request..."})

            # Generate SQL (fallback or LLM)
            try:
                sql = await generate_sql(session_id, question)
            except Exception as e:
                await ws.send_json({"type": "answer", "markdown": f"❗ Could not form SQL: {str(e)}"})
                continue

            # Execute SQL
            try:
                rows = await exec_sql(sql)
            except Exception as e:
                await ws.send_json({"type": "answer", "markdown": f"❗ Database error: {str(e)}"})
                continue

            # DO NOT send internal stream objects to frontend.
            # Instead, inform about number of rows (status), then send final answer and file info.
            await ws.send_json({"type": "status", "message": f"Fetched {len(rows)} rows."})

            # Prepare answer
            await ws.send_json({"type": "status", "message": "Preparing final answer..."})
            try:
                answer_md = await generate_answer(question, sql, rows)
            except Exception as e:
                answer_md = f"❗ Could not generate answer: {str(e)}"

            # Send the final answer
            await ws.send_json({"type": "answer", "markdown": answer_md})

            # Save excel and send file + preview (if rows exist)
            if rows:
                filename = f"query_{uuid.uuid4().hex[:8]}.xlsx"
                loop = asyncio.get_event_loop()
                try:
                    res = await loop.run_in_executor(None, lambda: write_styled_excel_and_preview(rows, filename))
                    await ws.send_json({"type": "file", "filename": res["filename"], "url": f"/downloads/{res['filename']}"})
                    await ws.send_json({"type": "file_preview", "file": res["filename"], "preview_rows": res["preview_rows"]})
                except Exception as e:
                    await ws.send_json({"type": "status", "message": f"Could not write file: {str(e)}"})

            # Persist memory: store Q/A pair in session memory (last N)
            try:
                async with session_lock:
                    session_memory.setdefault(session_id, []).append({"q": question, "a": answer_md})
                    # trim to last N
                    session_memory[session_id] = session_memory[session_id][-LAST_N_MEMORY:]
            except Exception:
                pass

    except WebSocketDisconnect:
        # On disconnect, clear session memory
        async with session_lock:
            try:
                if session_id in session_memory:
                    del session_memory[session_id]
            except Exception:
                pass
        return

# ----------------------------
# Root + startup
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "msg": "Diamond DB AI — Minimal Events backend running"}

@app.on_event("startup")
async def on_startup():
    try:
        await get_pg_pool()
    except Exception:
        pass
    try:
        await get_schema()
    except Exception:
        pass

@app.on_event("shutdown")
async def on_shutdown():
    global _pg_pool
    if _pg_pool:
        try:
            await _pg_pool.close()
        except Exception:
            pass
        _pg_pool = None
