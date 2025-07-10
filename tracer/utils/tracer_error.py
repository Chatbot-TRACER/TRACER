"""Custom exception types for the Tracer application.

This module defines a hierarchy of custom exception classes for the Tracer application.
This allows for more specific error handling and communication of issues,
both in the CLI and in a potential web interface.

Exception Hierarchy:
- TracerError: Base class for all custom exceptions in the application.
- GraphvizNotInstalledError: Raised when Graphviz is not installed on the system.
- ConnectorError: Base class for all chatbot connector-related errors.
  - ConnectorConnectionError: Raised when unable to establish connection to chatbot endpoint.
  - ConnectorAuthenticationError: Raised when chatbot connector authentication fails.
  - ConnectorConfigurationError: Raised when chatbot connector configuration is invalid.
  - ConnectorResponseError: Raised when chatbot connector receives invalid or unexpected responses.
- LLMError: Raised for errors related to the Language Model (LLM) API.
"""

# No additional imports required


class TracerError(Exception):
    """Custom exception for errors during Tracer execution."""


class GraphvizNotInstalledError(TracerError):
    """Raised when Graphviz is not installed."""


class ConnectorError(TracerError):
    """Base class for chatbot connector-related errors.

    This class captures rich context about the connector failure (URL, status
    code, etc.) but remains flexible - it accepts arbitrary keyword arguments
    so that caller sites can pass whatever information they have available
    without worrying about strict signatures.  All keyword arguments are
    stored as attributes on the instance for later inspection/logging.
    """

    def __init__(self, message: str | None = None, **kwargs: object) -> None:
        """Initialize ConnectorError.

        Args:
            message: Human-readable error message. If *None*, one will be
                synthesized from *kwargs*.
            **kwargs: Arbitrary keyword arguments giving additional context
                (e.g. ``url``, ``status_code``, ``connector_type``). These are
                attached as attributes to the exception instance for later
                inspection.
        """
        # Save all provided keyword arguments as attributes (e.g. url,
        # connector_type, status_code, original_error, response_text, ...).
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Build a default message if none was given.
        if message is None:
            message_parts = [
                kwargs.get("connector_type", "UnknownConnector"),
                "error",
            ]
            if "status_code" in kwargs:
                message_parts.append(f"(HTTP {kwargs['status_code']})")
            message = " ".join(message_parts)

        super().__init__(message)


class ConnectorConnectionError(ConnectorError):
    """Raised when unable to establish connection to chatbot endpoint."""


class ConnectorAuthenticationError(ConnectorError):
    """Raised when chatbot connector authentication fails."""


class ConnectorConfigurationError(ConnectorError):
    """Raised when chatbot connector configuration is invalid."""


class ConnectorResponseError(ConnectorError):
    """Raised when chatbot connector receives invalid or unexpected responses."""


class LLMError(TracerError):
    """Raised for errors related to the LLM API."""
