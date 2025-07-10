"""Custom exception types for the Tracer application.

This module defines a hierarchy of custom exception classes for the Tracer application.
This allows for more specific error handling and communication of issues,
both in the CLI and in a potential web interface.

Exception Hierarchy:
- TracerError: Base class for all custom exceptions in the application.
- GraphvizNotInstalledError: Raised when Graphviz is not installed on the system.
- ConnectorError: Raised for issues related to chatbot connectors (e.g., connection failures, invalid configurations).
- LLMError: Raised for errors related to the Language Model (LLM) API.
"""


class TracerError(Exception):
    """Custom exception for errors during Tracer execution."""


class GraphvizNotInstalledError(TracerError):
    """Raised when Graphviz is not installed."""


class ConnectorError(TracerError):
    """Raised for errors related to the chatbot connector."""


class LLMError(TracerError):
    """Raised for errors related to the LLM API."""
