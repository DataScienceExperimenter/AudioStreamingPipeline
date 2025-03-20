import sys
from typing import Optional

from loguru import logger

from src.utils.context import request_id_var, user_id_var, session_var


def context_formatter(record):
    """Format log messages with request context"""
    # Get context variables
    request_id = request_id_var.get()
    user_id = user_id_var.get()
    session = session_var.get()

    # Build context string
    context_parts = []
    if request_id:
        context_parts.append(f"request_id={request_id}")
    if user_id:
        context_parts.append(f"user_id={user_id}")
    if session:
        context_parts.append(f"session={session}")

    # Add context to record
    context_str = " ".join(context_parts)
    if context_str:
        record["extra"]["context"] = f"[{context_str}]"
    else:
        record["extra"]["context"] = ""

    return "{time} | {level: <8} | {extra[context]} {message}\\n"


# Configure logger
def configure_logger(log_level: str = "INFO", log_format: str = "console"):
    """Configure the logger with the specified settings"""
    # Remove default handler
    logger.remove()
    
    # Configure with context
    logger.configure(extra={"context": ""})
    
    # Add handler with our custom formatter
    logger.add(
        sys.stderr,
        format=context_formatter,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if needed
    if log_format == "file" or log_format == "both":
        logger.add(
            "logs/app.log",
            format=context_formatter,
            level=log_level,
            rotation="10 MB",
            retention="1 week"
        )


# Initialize with default settings
configure_logger()