"""Chatbot connectors package."""

from .core import (
    Chatbot,
    ChatbotConfig,
    ChatbotResponse,
    EndpointConfig,
    Headers,
    Payload,
    RequestMethod,
    ResponseProcessor,
    SimpleTextProcessor,
)
from .factory import ChatbotFactory
from .implementations.millionbot import ChatbotAdaUam, MillionBot, MillionBotConfig
from .implementations.rasa import RasaChatbot, RasaConfig
from .implementations.taskyto import ChatbotTaskyto, TaskytoConfig

# Register all chatbot implementations with the factory
ChatbotFactory.register_chatbot("taskyto", ChatbotTaskyto, requires_url=True, description="Taskyto chatbot connector")
ChatbotFactory.register_chatbot(
    "ada-uam", ChatbotAdaUam, requires_url=False, description="Pre-configured ADA UAM chatbot"
)
ChatbotFactory.register_chatbot("rasa", RasaChatbot, requires_url=True, description="RASA chatbot connector")

__all__ = [
    "Chatbot",
    "ChatbotAdaUam",
    "ChatbotConfig",
    "ChatbotFactory",
    "ChatbotResponse",
    "ChatbotTaskyto",
    "EndpointConfig",
    "Headers",
    "MillionBot",
    "MillionBotConfig",
    "Payload",
    "RasaChatbot",
    "RasaConfig",
    "RequestMethod",
    "ResponseProcessor",
    "SimpleTextProcessor",
    "TaskytoConfig",
]
