import sqlparse
from src.core.config import settings
from src.core.logging import get_logger
from typing import Tuple, Optional
import re

logger = get_logger(__name__)


class SQLValidator:
    DANGEROUS_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
        "MERGE", "REPLACE", "CALL", "LOCK", "UNLOCK",
    ]

    def __init__(self, allowed_table: str, allowed_schema: str = "public"):
        self.allowed_table = allowed_table
        self.allowed_schema = allowed_schema

    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        if not sql or not sql.strip():
            return False, "SQL query is empty"

        sql_clean = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql_clean = re.sub(r"/\*.*?\*/", "", sql_clean, flags=re.DOTALL)

        statements = [s.strip() for s in sql_clean.strip().split(";") if s.strip()]
        if len(statements) > 1:
            return False, "Multiple statements not allowed"

        try:
            _ = sqlparse.parse(sql_clean)[0]
        except Exception as e:
            return False, f"SQL parse error: {e}"

        sql_upper = sql_clean.upper()
        for kw in self.DANGEROUS_KEYWORDS:
            if re.search(rf"\b{kw}\b", sql_upper):
                return False, f"DANGEROUS KEYWORD: {kw}"

        if not sql_upper.strip().startswith("SELECT"):
            return False, "Only SELECT queries allowed"

        if self.allowed_table.lower() not in sql_clean.lower():
            return False, f"Query must reference table: {self.allowed_table}"

        return True, None

    def enforce_limit(self, limit: Optional[int] = None, sql: Optional[str] = None) -> str:
        """Ensure query has safe LIMIT."""
        if sql is None:
            return ""

        #  fix for Pylance "None not assignable to int"
        if re.search(r"LIMIT\s+\d+", sql, flags=re.IGNORECASE):
            sql = sql.rstrip().rstrip(";")
            

        return sql
