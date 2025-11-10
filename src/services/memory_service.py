from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.logging import get_logger
from src.llm.summarizer import ResultSummarizer
from src.core.config import settings
from typing import Optional

logger = get_logger(__name__)

class MemoryService:
    def __init__(self, session: AsyncSession, user_id: Optional[int], session_id: str):
        self.session = session
        self.user_id = user_id
        self.session_id = session_id
        self.summarizer = ResultSummarizer(session_id=f"mem_{session_id}")


    #  Long-Term Memory (user level)

    async def get_user_memory(self) -> str:
        """Retrieve stored long-term memory for a given user."""
        if not self.user_id:
            return ""
        res = await self.session.execute(
            text(f"""
                SELECT memory_summary 
                FROM {settings.SCHEMA}.user_memory 
                WHERE user_id = :uid LIMIT 1
            """),
            {"uid": self.user_id},
        )
        row = res.fetchone()
        return row[0] if row else ""

    async def update_user_memory(self, new_fact: str):
        """Append a new memory fact to the user's memory."""
        if not self.user_id:
            logger.debug("Guest mode — skipping user memory update")
            return

        existing = await self.get_user_memory()
        combined = f"{existing}\n{new_fact}" if existing else new_fact

        await self.session.execute(
            text(f"""
                INSERT INTO {settings.SCHEMA}.user_memory (user_id, memory_summary)
                VALUES (:uid, :summary)
                ON CONFLICT (user_id) 
                DO UPDATE SET memory_summary = :summary, updated_at = CURRENT_TIMESTAMP
            """),
            {"uid": self.user_id, "summary": combined},
        )
        await self.session.commit()
        logger.info(f" Updated long-term memory for user {self.user_id}")


    #  Session-Level Memory (chat context)

    async def get_session_summary(self) -> str:
        """Retrieve a concise summary of the active session."""
        res = await self.session.execute(
            text(f"""
                SELECT summary 
                FROM {settings.SCHEMA}.session_summaries 
                WHERE session_id = :sid LIMIT 1
            """),
            {"sid": self.session_id},
        )
        row = res.fetchone()
        return row[0] if row else ""

    async def update_session_summary(self, summary_text: str):
        """Insert or update summary for the current chat session."""
        await self.session.execute(
            text(f"""
                INSERT INTO {settings.SCHEMA}.session_summaries (session_id, user_id, summary)
                VALUES (:sid, NULL, :summary)
                ON CONFLICT (session_id, user_id) 
                DO UPDATE SET summary = :summary, updated_at = CURRENT_TIMESTAMP
            """),
            {"sid": self.session_id, "summary": summary_text},
        )
        await self.session.commit()
        logger.info(f" Session summary updated for {self.session_id}")


    #  Auto-Summarization (optional)

    async def summarize_session_context(self, conversation: list[str]) -> Optional[str]:
        """Generate a summary for current session using LLM summarizer."""
        try:
            summary = await self.summarizer.summarize_texts(conversation)
            if summary:
                await self.update_session_summary(summary)
                return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
        return None
