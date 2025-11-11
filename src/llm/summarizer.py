import asyncio
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
        self.session_id = session_id


    #  SQL + Data Summarization
  
    async def summarize(
        self, user_query: str, sql: str, columns: List[str], rows: List[tuple], total_rows: int
    ) -> str:
        """Summarizes SQL query results into natural language."""
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
"""
        try:
            logger.info(f"[Summarizer] Generating result summary for session={self.session_id}")
            summary = await asyncio.wait_for(self.llm_provider.generate_response(prompt), timeout=20)
            return (summary or "").strip()
        except asyncio.TimeoutError:
            logger.warning(f"[Summarizer]  Summarize timeout for session={self.session_id}")
            return f" Query executed successfully with {total_rows} rows."
        except Exception as e:
            logger.error(f"[Summarizer] Summarize error: {e}")
            return f" Query executed successfully with {total_rows} rows."

   
    #  Conversational Chat Reply (non-SQL)
  
    async def generate_chat_reply(self, user_input: str) -> str:
        """Generate friendly conversational (non-SQL) reply."""
        prompt = f"""
You are a friendly AI assistant that helps users explore a PostgreSQL database.
Respond naturally and conversationally to the following:
"{user_input}"
"""
        try:
            logger.info(f"[Summarizer] Generating chat reply for session={self.session_id}")

            # Timeout protection
            reply = await asyncio.wait_for(self.llm_provider.generate_response(prompt), timeout=10)

            if not reply or not isinstance(reply, str) or reply.strip() == "":
                return "Hey there! How can I assist you with your data today?"
            return reply.strip()

        except asyncio.TimeoutError:
            logger.warning(f"[Summarizer]  Chat reply timed out for session={self.session_id}")
            return "Hey there ! How can I assist you with your data today?"
        except Exception as e:
            logger.error(f"[Summarizer] Chat reply generation failed: {e}")
            return "Hey there! How can I assist you with your data today?"

   
    #  Memory / History Summarization
  
    async def summarize_texts(self, texts: List[str]) -> str:
        """Summarize chat history into short memory summary."""
        if not texts:
            return ""
        combined_text = "\n".join(texts)
        prompt = f"""
Summarize this chat conversation in 2–3 lines.
Focus on what the user asked, what the AI answered, and context.

Conversation:
{combined_text}

Summary:
"""
        try:
            summary = await asyncio.wait_for(self.llm_provider.generate_response(prompt), timeout=15)
            return (summary or "").strip()
        except asyncio.TimeoutError:
            logger.warning(f"[Summarizer] summarize_texts timeout for session={self.session_id}")
            return ""
        except Exception as e:
            logger.error(f"[Summarizer] summarize_texts() failed: {e}")
            return ""

    #  Basic Stats Helper

    def _calculate_stats(self, columns, rows):
        """Basic statistics to enrich summaries."""
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
