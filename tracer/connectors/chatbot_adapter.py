"""Adapter for using chatbot-connectors library in TRACER."""

from typing import Any

from chatbot_connectors import Chatbot, ChatbotFactory
from chatbot_connectors.exceptions import ConnectorError


def create_chatbot_connector(chatbot_type: str, **kwargs: dict[str, Any]) -> Chatbot:
    """Create a chatbot connector using the chatbot-connectors library.

    Args:
        chatbot_type: Type of chatbot (rasa, millionbot, taskyto)
        **kwargs: Configuration parameters for the chatbot

    Returns:
        Chatbot connector instance

    Raises:
        ConnectorError: If there's an issue creating the connector
    """
    try:
        return ChatbotFactory.create_chatbot(chatbot_type, **kwargs)
    except ValueError as e:
        msg = f"Failed to create {chatbot_type} connector: {e}"
        raise ConnectorError(msg) from e


def get_available_chatbot_types() -> list[str]:
    """Get list of available chatbot types.

    Returns:
        List of available chatbot type names
    """
    return ChatbotFactory.get_available_types()


def get_chatbot_parameters(chatbot_type: str) -> list[Any]:
    """Get parameters required for a chatbot type.

    Args:
        chatbot_type: Type of chatbot

    Returns:
        List of Parameter objects
    """
    return ChatbotFactory.get_chatbot_parameters(chatbot_type)
