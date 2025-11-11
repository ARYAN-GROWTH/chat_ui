from src.llm.provider import LLMProvider
from src.llm.validator import SQLValidator
from src.core.config import settings
from src.core.logging import get_logger
from src.db.schema_inspect import SchemaInspector
from src.db.column_synonyms import COLUMN_SYNONYMS  
from typing import Optional, List, Dict, Tuple, AsyncGenerator, Union
import re
from difflib import get_close_matches

logger = get_logger(__name__)


class SQLAgent:
    """Handles NL → SQL generation using LLM, synonym mapping, and context enrichment."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.llm_provider = LLMProvider()
        self.validator = SQLValidator(
            allowed_table=settings.TABLE_NAME,
            allowed_schema=settings.SCHEMA,
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.schema_context = SchemaInspector._cached_schema_text or ""
        self.safe_mode = True

   
    #  Core SQL Generation
    
    async def generate_sql(self, user_query: str) -> str:
        """Generate valid SQL intelligently using schema + synonyms + smart enrichment."""

        synonym_hints = "\n".join(
            [f"- {', '.join(v)} → {k}" for k, v in COLUMN_SYNONYMS.items()]
        )

        prompt = f"""
You are an expert PostgreSQL SQL assistant.
Generate a clean, correct SQL query for the user's request.

Database Schema:
{self.schema_context}

Valid Columns and Synonyms:
{synonym_hints}

Rules:
1. Use only columns from this schema.
2. NEVER invent new column names.
3. Always use item_no for product/item filters if present.
4. Use synonyms to map user phrases (e.g., "customer name" → "customer_name").
5. Return only the SQL query (no explanation).

User question: {user_query}

SQL Query:
"""
        response = await self.llm_provider.generate_response(prompt)
        sql = self._extract_sql(response)
        sql = await self._validate_columns(sql)
        sql = self._enforce_context_columns(sql, user_query)
        self._append_to_history(user_query, sql)
        logger.info(f"Final SQL generated: {sql}")
        return sql

 
    #  Streaming SQL Generation
    
    async def stream_generate_sql(
        self, user_query: str
    ) -> AsyncGenerator[Union[str, dict[str, str]], None]:
        """Stream SQL token-by-token using cached schema + synonyms."""
        synonym_hints = "\n".join(
            [f"- {', '.join(v)} → {k}" for k, v in COLUMN_SYNONYMS.items()]
        )

        prompt = f"""
You are an expert PostgreSQL SQL assistant.
Generate a SELECT query for the user's question.

Database Schema:
{self.schema_context}

Valid Columns and Synonyms:
{synonym_hints}

Rules:
- Use only valid table columns (see list above).
- Never invent column names.
- Return only SQL (no comments or explanations).

User question: {user_query}

SQL Query:
"""
        collected = ""
        async for piece in self.llm_provider.stream_response(prompt):
            collected += piece
            yield piece

        sql = self._extract_sql(collected)
        sql = await self._validate_columns(sql)
        sql = self._enforce_context_columns(sql, user_query)
        self._append_to_history(user_query, sql)
        yield {"__final_sql__": sql}

  
    #  Intent Classification
   
    async def simple_classify_intent(self, prompt: str) -> str:
        """LLM-based fallback for intent classification."""
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

  
    # Context Column Enrichment (Smart Version)
 
    def _enforce_context_columns(self, sql: str, user_query: str = "") -> str:
        """
        Dynamically include key context columns (like item_no, etc.)
        based on what the user asked for.
        """
        try:
            sql_lower = sql.lower()
            if not sql_lower.startswith("select"):
                return sql

            base_context_cols = ["item_no"]

            #  Keyword → Column groups
            keyword_map = {
                "company": ["company"]
            }

            query_lower = user_query.lower()
            selected_cols = set(base_context_cols)

            # Detect keywords in user question
            for keyword, cols in keyword_map.items():
                if keyword in query_lower:
                    selected_cols.update(cols)

            # Find selected columns in current SQL
            select_part = sql_lower.split("from")[0]
            missing_cols = [c for c in selected_cols if c not in select_part]

            if missing_cols:
                after_select = sql_lower.find("select") + len("select")
                extra = ", ".join(missing_cols)
                sql = sql[:after_select] + f" {extra}," + sql[after_select:]

            return sql.strip().rstrip(";") + ";"

        except Exception as e:
            logger.warning(f" Context column enrichment failed: {e}")
            return sql


    #  Column Validation + Fuzzy Correction
   
    async def _validate_columns(self, sql: str) -> str:
        """Ensure only valid schema columns are used. Auto-fix invalid ones."""
        available_columns = list(COLUMN_SYNONYMS.keys())

        match = re.search(r"SELECT(.*?)FROM", sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return sql

        selected = match.group(1)
        cols = [
            c.strip().split()[0]
            for c in re.split(r",|\s+", selected)
            if c and not c.upper().startswith(("DISTINCT", "*"))
        ]

        missing = [c for c in cols if c not in available_columns and c != "*"]
        if not missing:
            return sql

        fixed_sql = sql
        for miss in missing:
            suggestion = get_close_matches(miss, available_columns, n=1, cutoff=0.6)
            if suggestion:
                logger.warning(f" Replacing unknown column '{miss}' → '{suggestion[0]}'")
                fixed_sql = re.sub(rf"\b{miss}\b", suggestion[0], fixed_sql, flags=re.IGNORECASE)
            elif self.safe_mode:
                logger.warning(f" Dropping invalid column: {miss}")
                fixed_sql = fixed_sql.replace(miss, "")

        return fixed_sql


    #  Validation Wrapper
   
    def validate_and_fix_sql(self, sql: str) -> Tuple[bool, str, Optional[str]]:
        """Validate SQL and enforce safety (LIMIT, schema, etc.)."""
        is_valid, error = self.validator.validate(sql)
        if not is_valid:
            return False, sql, error
        fixed_sql = self.validator.enforce_limit(sql=sql)
        return True, fixed_sql, None


    #  Utility Helpers
    
    def _extract_sql(self, response: str) -> str:
        """Clean raw LLM output."""
        sql = re.sub(r"```sql|```", "", response, flags=re.IGNORECASE)
        sql = re.sub(r"^(SQL Query:|Query:|Answer:)\s*", "", sql, flags=re.IGNORECASE).strip()
        lines = [line.strip() for line in sql.splitlines() if line.strip() and not line.startswith("--")]
        sql = " ".join(lines).strip()
        if not sql.endswith(";"):
            sql += ";"
        return sql

    def _append_to_history(self, user_query: str, sql: str):
        """Save chat + SQL in short-term memory."""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": f"SQL: {sql}"})
