import json
import logging
import sys
import time
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Simple structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = getattr(record, "request_id")
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = getattr(record, "latency_ms")
        if hasattr(record, "method"):
            log_data["method"] = getattr(record, "method")
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def get_logger(name: str = "nexus", level: str = "INFO") -> logging.Logger:
    """Get or configure logger with structured console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
