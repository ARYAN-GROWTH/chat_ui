from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import settings
from src.core.logging import get_logger
from typing import Dict, List, Optional

logger = get_logger(__name__)


class SchemaInspector:
    """
    Helper to inspect and cache database schema information for LLM prompts.

    - Caches table column metadata (in-memory) to avoid repeated DB roundtrips.
    - Caches a human-readable schema description used for LLM context.
    """

    # Editor-friendly declarations so static checkers (Pylance) know these attrs exist
    _schema_cache: Optional[Dict[str, str]] = None
    _cached_schema_text: Optional[str] = None

    def __init__(self, session: AsyncSession):
        self.session = session
        self.schema = settings.SCHEMA
        self.table_name = settings.TABLE_NAME

    async def get_table_schema(self) -> Dict[str, str]:
        """
        Return a mapping of column_name -> type (and length if available) for the configured table.
        Caches result in-memory on first fetch.
        """
        if SchemaInspector._schema_cache is not None:
            return SchemaInspector._schema_cache

        try:
            logger.info(f"Fetching schema for {self.schema}.{self.table_name} from DB...")
            query = text(
                """
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table_name
                ORDER BY ordinal_position;
                """
            )
            result = await self.session.execute(query, {"schema": self.schema, "table_name": self.table_name})
            columns: Dict[str, str] = {}
            for row in result:
                name = row[0]
                dtype = row[1]
                maxlen = row[2]
                columns[name] = f"{dtype}({maxlen})" if maxlen else dtype

            SchemaInspector._schema_cache = columns
            logger.info(f"Schema loaded: {len(columns)} columns")
            return columns

        except Exception as e:
            logger.error(f"Error fetching table schema: {e}")
            raise

    async def get_sample_rows(self, limit: int = 5) -> List[Dict]:
        """
        Return a list of sample rows (as dicts) from the configured table.
        Non-fatal: on error returns empty list and logs the issue.
        """
        try:
            query = text(f"SELECT * FROM {self.schema}.{self.table_name} LIMIT :limit")
            result = await self.session.execute(query, {"limit": limit})
            rows = result.fetchall()
            cols = result.keys()
            return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching sample rows: {e}")
            return []

    async def get_schema_description(self) -> str:
        """
        Produce a human-readable schema + small sample data description string suitable
        for providing to an LLM as context. This result is cached after the first call.
        """
        if SchemaInspector._cached_schema_text:
            return SchemaInspector._cached_schema_text

        try:
            schema = await self.get_table_schema()
            samples = await self.get_sample_rows(limit=3)

            description_lines = [f"Table: {self.schema}.{self.table_name}", "", "Columns:"]
            for col, typ in schema.items():
                description_lines.append(f"  - {col} ({typ})")

            if samples:
                description_lines.append("")
                description_lines.append("Sample data (first rows):")
                for i, row in enumerate(samples, 1):
                    description_lines.append(f"  Row {i}: {row}")

            SchemaInspector._cached_schema_text = "\n".join(description_lines)
            logger.info("Schema description cached for LLM context.")
            return SchemaInspector._cached_schema_text

        except Exception as e:
            logger.error(f"Failed to build schema description: {e}")
            raise
