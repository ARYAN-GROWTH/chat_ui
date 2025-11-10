from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SCHEMA: str = os.getenv("SCHEMA", "public")
    TABLE_NAME: str = os.getenv("TABLE_NAME", "dev_diamond2")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")


    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")

settings = Settings()
