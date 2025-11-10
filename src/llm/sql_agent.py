from src.llm.provider import LLMProvider
from src.llm.validator import SQLValidator
from src.core.config import settings
from src.core.logging import get_logger
from src.db.schema_inspect import SchemaInspector
from typing import Optional, List, Dict, Tuple, AsyncGenerator, Union
import re

logger = get_logger(__name__)


class SQLAgent:
    """Handles NL → SQL generation using LLM + cached schema + rule-based shortcuts."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.llm_provider = LLMProvider()
        self.validator = SQLValidator(
            allowed_table=settings.TABLE_NAME,
            allowed_schema=settings.SCHEMA,
        )
        self.conversation_history: List[Dict[str, str]] = []


        self.schema_context = SchemaInspector._cached_schema_text or ""

    async def generate_sql(self, user_query: str) -> str:
        """Generate SQL intelligently using rules + cached schema."""
        #  Rule-based shortcut for faster responses
        item_match = re.search(r"[A-Z0-9]{3,}[_-][A-Z0-9.]+", user_query)
        if item_match and "customer" in user_query.lower():
            item_code = item_match.group(0)
            sql = f"""
            SELECT DISTINCT item_no, customer_name
            FROM {settings.SCHEMA}.{settings.TABLE_NAME}
            WHERE item_no = '{item_code}';
            """
            self._append_to_history(user_query, sql.strip())
            logger.info(f" Rule-based SQL generated: {sql.strip()}")
            return sql.strip()

        #  Fallback — LLM-based generation
        prompt = f"""You are an expert PostgreSQL assistant.
You know this database schema:
{self.schema_context}

Generate a clean SELECT SQL query for the user's request.

Rules:
1. Only use table: {settings.SCHEMA}.{settings.TABLE_NAME}
2. Never modify data (SELECT only)
3. Return only SQL, no explanation.

User query: {user_query}

SQL Query:
"""
        response = await self.llm_provider.generate_response(prompt)
        sql = self._extract_sql(response)
        self._append_to_history(user_query, sql)
        logger.info(f"LLM-generated SQL: {sql}")
        return sql


    async def stream_generate_sql(
        self, user_query: str
    ) -> AsyncGenerator[Union[str, dict[str, str]], None]:
        """Stream SQL token-by-token using cached schema context."""
        prompt = f"""You are an expert PostgreSQL SQL assistant.
Database schema:
{self.schema_context}

Generate a SELECT query for the user's question.
Rules:
- Only use table {settings.SCHEMA}.{settings.TABLE_NAME}
- Return only SQL.

User question: {user_query}

SQL Query:
"""
        collected = ""
        async for piece in self.llm_provider.stream_response(prompt):
            collected += piece
            yield piece

        sql = self._extract_sql(collected)
        self._append_to_history(user_query, sql)
        yield {"__final_sql__": sql}


    async def simple_classify_intent(self, prompt: str) -> str:
        """
        Fallback intent classifier using LLM or simple keyword heuristics.
        Returns one of: 'SQL_QUERY', 'CHAT', 'OTHER'.
        """
        try:
            response = await self.llm_provider.generate_response(prompt)
            clean_resp = (response or "").strip().upper()
            if "SQL" in clean_resp:
                return "SQL_QUERY"
            if "CHAT" in clean_resp:
                return "CHAT"
            return "OTHER"
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "SQL_QUERY"


    def _extract_sql(self, response: str) -> str:
        """Extracts clean SQL from LLM response."""
        sql = re.sub(r"```sql|```", "", response, flags=re.IGNORECASE)
        sql = re.sub(r"^(SQL Query:|Query:|Answer:)\s*", "", sql, flags=re.IGNORECASE).strip()
        lines = [line.strip() for line in sql.splitlines() if line.strip() and not line.startswith("--")]
        sql = " ".join(lines).strip()
        if not sql.endswith(";"):
            sql += ";"
        return sql

    def validate_and_fix_sql(self, sql: str) -> Tuple[bool, str, Optional[str]]:
        """Validate query and ensure safe LIMIT usage."""
        is_valid, error = self.validator.validate(sql)
        if not is_valid:
            return False, sql, error
        fixed_sql = self.validator.enforce_limit(sql=sql)
        return True, fixed_sql, None

    def _append_to_history(self, user_query: str, sql: str):
        """Save query + generated SQL in local memory for context continuity."""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": f"SQL: {sql}"})
