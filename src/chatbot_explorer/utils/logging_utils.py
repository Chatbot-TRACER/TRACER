"""Logging utilities for the chatbot explorer."""

import logging
import sys

LOGGER_NAME = "chatbot_explorer"

# Define custom VERBOSE log level number (between INFO and DEBUG)
VERBOSE_LEVEL_NUM = 15
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")


def verbose(self: logging.Logger, message: str, *args: object, **kws: any) -> None:
    """Logs a message with level VERBOSE on this logger.

    Args:
        self (logging.Logger): The logger instance.
        message (str): The message to log.
        *args: Variable length argument list.
        **kws: Arbitrary keyword arguments.
    """
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(VERBOSE_LEVEL_NUM, message, args, **kws)


logging.Logger.verbose = verbose


class ConditionalFormatter(logging.Formatter):
    """Applies different formats based on log level.

    INFO and VERBOSE levels get minimal formatting (message only).
    DEBUG, WARNING, ERROR, CRITICAL levels get detailed formatting.
    """

    SIMPLE_FORMAT = "%(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        """Initializes the ConditionalFormatter."""
        super().__init__(fmt=self.SIMPLE_FORMAT, datefmt=self.DATE_FORMAT)
        # Create formatter instances for each style *once* during initialization
        self._simple_formatter = logging.Formatter(self.SIMPLE_FORMAT, datefmt=self.DATE_FORMAT)
        self._detailed_formatter = logging.Formatter(self.DETAILED_FORMAT, datefmt=self.DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record based on its level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message string.
        """
        # Process any escape sequences in the message
        if isinstance(record.msg, str):
            record.msg = record.msg.encode("utf-8").decode("unicode_escape")

        # Use the simple format for INFO and VERBOSE levels
        if record.levelno in (logging.INFO, VERBOSE_LEVEL_NUM):
            return self._simple_formatter.format(record)
        # Use the detailed format for other levels
        return self._detailed_formatter.format(record)


def setup_logging(verbosity: int = 0) -> None:
    """Configures the application's named logger (`chatbot_explorer`).

    Sets the logger's threshold level and adds a handler with a
    ConditionalFormatter that mimics print for INFO/VERBOSE levels and
    provides details for DEBUG/WARNING/ERROR levels.

    Args:
        verbosity (int): Controls the logging threshold:
                         0=INFO, 1=VERBOSE, 2+=DEBUG.
    """
    if verbosity == 0:
        log_level = logging.INFO
    elif verbosity == 1:
        log_level = VERBOSE_LEVEL_NUM
    else:  # verbosity >= 2
        log_level = logging.DEBUG

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # Clear previous handlers to prevent duplication if setup is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)  # Log to standard output
    formatter = ConditionalFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent messages from propagating to the root logger
    # This is important to avoid double logging if root has default handlers
    logger.propagate = False

    # Use standard logging format strings for efficiency
    logger.debug("Chatbot Explorer logging configured to level: %s", logging.getLevelName(log_level))


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Retrieves the application logger or a child logger.

    Ensures that the retrieved logger inherits the configuration set by `setup_logging`.

    Args:
        name (str): The name of the logger. Defaults to the main application logger.
                    Can be used to get child loggers like 'chatbot_explorer.agent'.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)
