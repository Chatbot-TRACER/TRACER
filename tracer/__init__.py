"""Chatbot Explorer package, interacts with a chatbot to find its functionalities and limitations and creates user profiles based on that."""

from .main import main
from .utils.tracer_error import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorResponseError,
    GraphvizNotInstalledError,
    LLMError,
    TracerError,
)

__all__ = [
    "ConnectorAuthenticationError",
    "ConnectorConfigurationError",
    "ConnectorConnectionError",
    "ConnectorError",
    "ConnectorResponseError",
    "GraphvizNotInstalledError",
    "LLMError",
    "TracerError",
    "main",
]
