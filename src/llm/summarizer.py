from src.llm.provider import LLMProvider
from src.core.logging import get_logger
from src.db.schema_inspect import SchemaInspector
from src.core.config import settings
from typing import List
import json

logger = get_logger(__name__)


class ResultSummarizer:
    def __init__(self, session_id: str = "summarizer"):
        
        schema_context = SchemaInspector._cached_schema_text or "(schema not loaded)"
        self.llm_provider = LLMProvider(
            system_message=f"""
You are a smart data analysis assistant that summarizes SQL query results into short,
clear, business-friendly summaries for the table {settings.SCHEMA}.{settings.TABLE_NAME}.

Here is the database schema you can refer to when understanding results:
{schema_context}

Rules:
1. Be concise (2–3 lines max).
2. Mention relevant column names and relationships.
3. Avoid SQL jargon — write natural English.
4. If the question mentions a specific item or customer, highlight it.
5. If the data looks repetitive or limited, say that clearly.
"""
        )

    #  Core: SQL + Data Summarization (Schema-aware)

    async def summarize(
        self, user_query: str, sql: str, columns: List[str], rows: List[tuple], total_rows: int
    ) -> str:
        """Summarizes SQL query results (structured tabular data) into natural language."""
        if not rows:
            return "No results found."

        sample_data = [dict(zip(columns, r)) for r in rows[:10]]
        stats = self._calculate_stats(columns, rows)
        schema_context = SchemaInspector._cached_schema_text or ""

        prompt = f"""
The following SQL query and its result were executed.

User Question: {user_query}
SQL Query: {sql}
Table Schema: {schema_context}

Total Rows: {total_rows}
Columns: {', '.join(columns)}

Sample Data (first rows):
{json.dumps(sample_data, indent=2, default=str)}

{stats}

Write a clear, 2–3 line business summary.
Example format:
- If user asked about an item or customer, mention that directly.
- Describe patterns, unique values, and counts if relevant.
- Use human-friendly language.

Summary:
"""
        try:
            summary = await self.llm_provider.generate_response(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"Summarize error: {e}")
            return f"Query returned {total_rows} rows."

 
    #  Conversational Reply Generator

    async def generate_chat_reply(self, user_input: str) -> str:
        """Generate conversational (non-SQL) reply."""
        prompt = f"""
You are a friendly AI assistant that helps users query a PostgreSQL database.
Respond to this message naturally, as if you are chatting:
"{user_input}"
"""
        try:
            return await self.llm_provider.generate_response(prompt)
        except Exception as e:
            logger.error(f"Chat reply generation failed: {e}")
            return "I'm here to help you explore your data. Could you clarify your question?"

# Memory / History Summarization

    async def summarize_texts(self, texts: List[str]) -> str:
        """Summarize chat messages into a short memory summary."""
        if not texts:
            return ""
        combined_text = "\n".join(texts)
        prompt = f"""
Summarize the following chat or conversation into a short, 2–3 line memory summary.
Focus on what the user wanted, what the AI answered, and main context.

Conversation:
{combined_text}

Summary:
"""
        try:
            summary = await self.llm_provider.generate_response(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"summarize_texts() failed: {e}")
            return ""


    def _calculate_stats(self, columns, rows):
        """Basic stats to enrich summaries."""
        if not rows:
            return ""
        text = "\nBasic statistics:\n"
        for i, col in enumerate(columns):
            try:
                vals = [r[i] for r in rows if r[i] is not None]
                if vals and isinstance(vals[0], (int, float)):
                    text += f"  - {col}: min={min(vals)}, max={max(vals)}, avg={sum(vals)/len(vals):.2f}\n"
                elif len(set(vals)) <= 10:
                    unique_values = ", ".join(map(str, set(vals)))
                    text += f"  - {col}: {len(set(vals))} unique values ({unique_values})\n"
            except Exception:
                continue
        return text
