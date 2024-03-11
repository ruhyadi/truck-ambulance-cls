import rootutils

ROOT = rootutils.autosetup()

import logging
from logging.handlers import TimedRotatingFileHandler

from colorlog import ColoredFormatter


def get_logger(level: int = 20) -> logging:
    """
    Get inference logger.

    Args:
        level (int, optional): logging level. Defaults to None.

    Returns:
        logging: logger function

    Examples:
        >>> logger = get_logger()
        >>> logger.debug("debug message")
        2023-01-03 12:00:00,000 [DEBUG] debug message | my_app.py:23
        >>> logger.info("info message")
        2023-01-03 12:00:00,000 [INFO] info message | my_app.py:23
    """
    # rotating file handler to save logging file and
    # rotate every midnight and keep 30 days of logs
    rfh = TimedRotatingFileHandler(
        ROOT / "logs/logs.log",
        when="midnight",
        backupCount=3,
    )
    rfh.setLevel(level)
    rfh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s | %(filename)s:%(lineno)d"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColoredFormatter(
            "%(cyan)s%(asctime)s%(reset)s %(log_color)s[%(levelname)s] %(message)s | %(filename)s:%(lineno)d %(name)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold,red",
            },
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[rfh, console_handler],
    )

    return logging
