from openai import AsyncOpenAI
from src.core.config import settings
from src.core.logging import get_logger
from typing import AsyncGenerator, Optional

logger = get_logger(__name__)

if not settings.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set in .env. LLM calls will fail if used.")

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


class LLMProvider:
    """Simple wrapper around OpenAI Async client for streaming and non-streaming."""

    def __init__(self, model: Optional[str] = None, system_message: Optional[str] = None):
        self.model: str = model or settings.DEFAULT_MODEL
        self.system_message: str = (
            system_message
            or "You are a helpful assistant that generates PostgreSQL SELECT queries and summaries."
        )

    async def generate_response(self, prompt: str) -> str:
        """Non-streaming LLM call."""
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
            )

            # ✅ access message content directly (no .get)
            content = response.choices[0].message.content  # type: ignore
            return (content or "").strip()

        except Exception as e:
            logger.error(f"LLM generate_response error: {e}")
            raise

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streaming LLM call — yields incremental tokens."""
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                max_tokens=1024,
                temperature=0.0,
            )

            async for chunk in stream:
                # ✅ each chunk contains delta with .content
                delta = chunk.choices[0].delta  # type: ignore
                text_piece = getattr(delta, "content", None)
                if text_piece:
                    yield text_piece

        except Exception as e:
            logger.error(f"LLM stream_response error: {e}")
            yield f"[LLM STREAM ERROR] {e}"
