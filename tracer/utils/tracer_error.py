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


class TracerError(Exception):
    """Custom exception for errors during Tracer execution."""


class GraphvizNotInstalledError(TracerError):
    """Raised when Graphviz is not installed."""


class ConnectorError(TracerError):
    """Base class for chatbot connector-related errors."""


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
