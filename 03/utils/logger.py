import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a stdout logger with the shared format.

    Args:
        name: Logger name, usually ``__name__``.
        level: Minimum level of emitted records.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    # Jupyter re-imports modules; add the handler only once to avoid duplicated lines.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
