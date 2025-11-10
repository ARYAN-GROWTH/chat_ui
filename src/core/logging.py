from loguru import logger
import sys
import os

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, "app_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="7 days",
    level="ERROR"
)

def get_logger(name: str):
    return logger.bind(name=name)
