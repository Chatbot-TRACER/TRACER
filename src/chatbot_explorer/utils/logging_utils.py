"""Logging utilities for the chatbot explorer."""

import logging
import re
import sys

LOGGER_NAME = "chatbot_explorer"

# Define custom VERBOSE log level number (between INFO and DEBUG)
VERBOSE_LEVEL_NUM = 15
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")

try:
    import colorama

    colorama.init(autoreset=True)  # Automatically reset color after each print

    C_RESET = colorama.Style.RESET_ALL
    C_INFO_HEADER = colorama.Fore.GREEN + colorama.Style.BRIGHT  # Green for phase starts/ends
    C_VERBOSE_SPEAKER1 = colorama.Fore.CYAN  # Cyan for Explorer
    C_VERBOSE_SPEAKER2 = colorama.Fore.MAGENTA  # Magenta for Chatbot
    C_WARNING = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    C_ERROR = colorama.Fore.RED + colorama.Style.BRIGHT
    C_DEBUG = colorama.Fore.BLUE  # Blue for debug details
except ImportError:
    print("Warning: colorama not found. Proceeding without colored logs.", file=sys.stderr)
    C_RESET = C_INFO_HEADER = C_VERBOSE_SPEAKER1 = C_VERBOSE_SPEAKER2 = ""
    C_WARNING = C_ERROR = C_DEBUG = ""


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
    """Applies different formats and colors based on log level.

    - INFO/VERBOSE: Minimal format. INFO headers colored green. VERBOSE speakers colored.
    - DEBUG: Detailed format, colored blue.
    - WARNING: Detailed format, colored yellow.
    - ERROR/CRITICAL: Detailed format, colored red.
    """

    SIMPLE_FORMAT = "%(message)s"
    DETAILED_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Precompile regex for speaker detection (case-insensitive)
    SPEAKER1_PATTERN = re.compile(r"^\s*Explorer:", re.IGNORECASE)
    SPEAKER2_PATTERN = re.compile(r"^\s*Chatbot:", re.IGNORECASE)
    # Things that start with three dashes
    HEADER_PATTERN = re.compile(r"^\s*---+", re.IGNORECASE)

    def __init__(self) -> None:
        """Initializes the ConditionalFormatter."""
        super().__init__(fmt=self.SIMPLE_FORMAT, datefmt=self.DATE_FORMAT)
        self._simple_formatter = logging.Formatter(self.SIMPLE_FORMAT, datefmt=self.DATE_FORMAT)
        self._detailed_formatter = logging.Formatter(self.DETAILED_FORMAT, datefmt=self.DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record based on its level, adding color.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted and potentially colored log message string.
        """
        # First, format the message using the base class or specific formatters
        if record.levelno in (logging.INFO, VERBOSE_LEVEL_NUM):
            log_message = self._simple_formatter.format(record)
            # Apply colors based on content for simple formats
            if record.levelno == logging.INFO and self.HEADER_PATTERN.match(log_message):
                log_fmt = f"{C_INFO_HEADER}{log_message}{C_RESET}"
            elif record.levelno == VERBOSE_LEVEL_NUM:
                if self.SPEAKER1_PATTERN.match(log_message):
                    log_fmt = f"{C_VERBOSE_SPEAKER1}{log_message}{C_RESET}"
                elif self.SPEAKER2_PATTERN.match(log_message):
                    log_fmt = f"{C_VERBOSE_SPEAKER2}{log_message}{C_RESET}"
                else:
                    log_fmt = log_message  # Default color for other VERBOSE messages
            else:
                log_fmt = log_message  # Default color for other INFO messages

        elif record.levelno == logging.DEBUG:
            log_fmt = f"{C_DEBUG}{self._detailed_formatter.format(record)}{C_RESET}"
        elif record.levelno == logging.WARNING:
            log_fmt = f"{C_WARNING}{self._detailed_formatter.format(record)}{C_RESET}"
        elif record.levelno >= logging.ERROR:  # ERROR and CRITICAL
            log_fmt = f"{C_ERROR}{self._detailed_formatter.format(record)}{C_RESET}"
        else:  # Other levels (use detailed by default)
            log_fmt = self._detailed_formatter.format(record)

        return log_fmt  # No final strip needed as colorama resets


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
