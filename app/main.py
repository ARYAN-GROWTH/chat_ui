import os
import re
import json
import time
import uuid
import asyncio
import datetime
from typing import Any, Dict, List, Optional
from openpyxl.worksheet.worksheet import Worksheet
from typing import cast
import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from langchain_openai import ChatOpenAI
import logging
from logging.handlers import RotatingFileHandler



LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("diamond-ai")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(f"{LOG_DIR}/server.log", maxBytes=5_000_000, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
))


logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(" Server Booting...")



load_dotenv()

def env(key: str):
    val = os.getenv(key)
    if not val:
        logger.error(f"Missing environment variable: {key}")
        raise RuntimeError(f"{key} missing")
    return val

OPENAI_API_KEY = env("OPENAI_API_KEY")
DATABASE_URL_ASYNC = env("DATABASE_URL_ASYNC")

SCHEMA = os.getenv("SCHEMA", "public")
TABLE_NAME = os.getenv("TABLE_NAME", "dev_diamond2")

SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL", 600))
LAST_N_MEMORY = int(os.getenv("LAST_N_MEMORY", 3))

SQL_MODEL_NAME = os.getenv("SQL_MODEL_NAME", "gpt-4.1-mini")
ANSWER_MODEL_NAME = os.getenv("ANSWER_MODEL_NAME", "gpt-4o-mini")

USER_HOME = os.path.expanduser("~")
DOWNLOAD_DIR = os.path.join(USER_HOME, "Downloads", "download")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DUMPS_DIR = os.path.join(os.getcwd(), "dumps")
os.makedirs(DUMPS_DIR, exist_ok=True)


app = FastAPI(title="Diamond DB AI Backend — Final")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")



@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.exception(f" Unhandled Error: {exc}")
    return JSONResponse({"error": "Internal Server Error"}, status_code=500)


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    logger.error(f"Validation Error: {exc}")
    return JSONResponse({"error": "Invalid Input"}, status_code=422)


_pg_pool: Optional[asyncpg.pool.Pool] = None
_schema_cache: Dict[str, Any] = {"value": None, "expires_at": 0}
session_memory: Dict[str, List[Dict[str, Any]]] = {}
session_lock = asyncio.Lock()



SQL_MODEL = ChatOpenAI(model=SQL_MODEL_NAME, temperature=0)
ANSWER_MODEL = ChatOpenAI(model=ANSWER_MODEL_NAME, temperature=0)

def safe_to_text(resp: Any) -> str:
    try:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, list):
            return " ".join(safe_to_text(x) for x in resp)
        if isinstance(resp, dict):
            return str(resp.get("content") or "")
        return str(resp)
    except Exception:
        return ""


def serialize_value(val):
    if isinstance(val, datetime.datetime):
        return val.isoformat()
    if isinstance(val, datetime.date):
        return val.isoformat()
    return val


def serialize_row(row: Dict[str, Any]):
    return {k: serialize_value(v) for k, v in row.items()}



async def get_pg_pool():
    global _pg_pool
    if _pg_pool is None:
        logger.info("🔌 Creating PostgreSQL Pool...")
        _pg_pool = await asyncpg.create_pool(DATABASE_URL_ASYNC, min_size=1, max_size=10)
    return _pg_pool


async def get_schema():
    now = time.time()
    if _schema_cache["value"] and _schema_cache["expires_at"] > now:
        return _schema_cache["value"]

    pool = await get_pg_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema=$1 AND table_name=$2
            ORDER BY ordinal_position
            """,
            SCHEMA, TABLE_NAME
        )

    columns = [dict(r) for r in rows]
    _schema_cache.update({"value": columns, "expires_at": now + SCHEMA_CACHE_TTL})

    return columns


def detect_intent_fallback(question: str, schema_cols: List[Dict[str, Any]]):
    q = question.lower()

    if "database" in q or "columns" in q:
        cols = ", ".join(f"\"{c['column_name']}\"" for c in schema_cols[:20])
        return f"SELECT {cols} FROM {SCHEMA}.{TABLE_NAME} LIMIT 1"

    if "main category" in q:
        return f"SELECT DISTINCT \"main_category\" FROM {SCHEMA}.{TABLE_NAME} LIMIT 200"

    if "subcategory" in q:
        return f"SELECT DISTINCT \"subcategory\" FROM {SCHEMA}.{TABLE_NAME} LIMIT 200"

    return None



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


def sanitize_sql(raw: str):
    txt = raw.strip().replace("```sql", "").replace("```", "")
    m = re.search(r"(select[\s\S]*?)(;|$)", txt, flags=re.I)
    if not m:
        raise ValueError("No SELECT found.")
    return m.group(1).strip()



async def generate_sql(session_id: str, question: str) -> str:
    try:
        schema_cols = await get_schema()
    except:
        return "SELECT 1"

    fallback = detect_intent_fallback(question, schema_cols)
    if fallback:
        return fallback

    async with session_lock:
        mem = session_memory.get(session_id, [])

    memory_text = "; ".join(f"Q:{p['q']} A:{p['a']}" for p in mem[-LAST_N_MEMORY:])

    prompt = SQL_PROMPT_TEMPLATE.format(
        schema=SCHEMA,
        table=TABLE_NAME,
        columns=", ".join(c["column_name"] for c in schema_cols),
        memory=memory_text,
        question=question
    )

    try:
        resp = await asyncio.to_thread(SQL_MODEL.invoke, prompt)
        return sanitize_sql(safe_to_text(resp))
    except:
        return "SELECT 1"



async def exec_sql(sql: str):
    pool = await get_pg_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute("SET statement_timeout='10000'")
            rows = await conn.fetch(sql)
            return [dict(r) for r in rows]
    except:
        return []



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



async def generate_answer(question, sql, rows):
    prompt = ANSWER_PROMPT.format(
        question=question,
        sql=sql,
        rows=json.dumps(rows[:10], default=str)
    )

    try:
        resp = await asyncio.to_thread(ANSWER_MODEL.invoke, prompt)
        return safe_to_text(resp)
    except:
        return "Could not generate answer."



def try_parse_date(value):
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return None


def write_styled_excel_and_preview(rows, filename):
    file_path = os.path.join(DOWNLOAD_DIR, filename)

    wb = Workbook()
    

    sheet = cast(Worksheet, wb.active)
    sheet.title = "Results"

    if rows:
        headers = list(rows[0].keys())

        # Header
        for idx, h in enumerate(headers, 1):
            c = sheet.cell(row=1, column=idx, value=h)
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1E4F9C")
            c.alignment = Alignment(horizontal="center")

        # Data
        for r_i, row in enumerate(rows, start=2):
            for c_i, h in enumerate(headers, start=1):
                v = row.get(h)
                d = try_parse_date(v)
                sheet.cell(row=r_i, column=c_i, value=d if d else v)

        # Auto-width
        for col_i in range(1, len(headers) + 1):
            max_len = 10
            for cell in sheet[get_column_letter(col_i)]:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            sheet.column_dimensions[get_column_letter(col_i)].width = max_len + 4

    wb.save(file_path)

    return {
        "filename": filename,
        "preview_rows": [serialize_row(r) for r in rows[:8]]
    }



GREETINGS = {"hello", "hi", "hey", "hola", "namaste", "hlo"}

@app.websocket("/ws/query")
async def ws_query(ws: WebSocket):
    await ws.accept()
    session_id = uuid.uuid4().hex
    logger.info(f" Client Connected: {session_id}")

    async with session_lock:
        session_memory[session_id] = []

    try:
        while True:
            raw = await ws.receive_text()

            try:
                payload = json.loads(raw)
            except:
                await ws.send_json({"type": "status", "message": "Invalid JSON"})
                continue

            question = safe_to_text(payload.get("question", ""))
            if not question:
                await ws.send_json({"type": "status", "message": "Ask a valid question"})
                continue

            if question.lower() in GREETINGS:
                await ws.send_json({"type": "answer", "message": " Hello! How can I help?"})
                continue

            await ws.send_json({"type": "status", "message": "Working..."})

            sql = await generate_sql(session_id, question)
            rows = await exec_sql(sql)

            await ws.send_json({"type": "status", "message": f"Fetched {len(rows)} rows"})

            answer = await generate_answer(question, sql, rows)
            await ws.send_json({"type": "answer", "message": answer})

            if rows:
                filename = f"query_{uuid.uuid4().hex[:6]}.xlsx"
                try:
                    res = write_styled_excel_and_preview(rows, filename)
                    await ws.send_json({
                        "type": "file",
                        "filename": filename,
                        "url": f"/downloads/{filename}"
                    })
                    await ws.send_json({
                        "type": "file_preview",
                        "preview_rows": res["preview_rows"]
                    })
                except Exception as e:
                    logger.error(f"Excel error: {e}")

            # Save memory
            async with session_lock:
                session_memory[session_id].append({"q": question, "a": answer})
                session_memory[session_id] = session_memory[session_id][-LAST_N_MEMORY:]

    except WebSocketDisconnect:
        logger.info(f" Client Disconnected: {session_id}")
        async with session_lock:
            session_memory.pop(session_id, None)



@app.get("/")
async def root():
    return {"status": "ok", "msg": "Diamond AI Backend Running"}


@app.on_event("startup")
async def startup():
    logger.info(" Server Started")


@app.on_event("shutdown")
async def shutdown():
    logger.info(" Server Shutdown Started")
    global _pg_pool
    if _pg_pool:
        await _pg_pool.close()
        logger.info(" PostgreSQL Pool Closed")
        _pg_pool = None