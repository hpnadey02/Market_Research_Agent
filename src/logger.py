# ==============================================================================
# logger.py — Structured Rotating Logger
# ==============================================================================
# Provides a centralized logger with both console and file output.
# Log files are saved in the project's logs/ directory with daily rotation.
# ==============================================================================

import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from src.config import LOGS_DIR


def get_logger(name: str = "MarketResearch") -> logging.Logger:
    """
    Returns a configured logger instance with console + rotating file handlers.
    
    Args:
        name: Logger namespace identifier.
    
    Returns:
        logging.Logger with structured formatting.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Formatter ────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── Rotating File Handler (10 MB, keep 5 backups) ────────────────────────
    log_file = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


