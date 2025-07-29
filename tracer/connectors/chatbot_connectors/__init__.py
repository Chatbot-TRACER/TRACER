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
from .implementations.millionbot import MillionBot
from .implementations.rasa import RasaChatbot, RasaConfig
from .implementations.taskyto import ChatbotTaskyto, TaskytoConfig

# Register all chatbot implementations with the factory
ChatbotFactory.register_chatbot("taskyto", ChatbotTaskyto, description="Taskyto chatbot connector")
ChatbotFactory.register_chatbot("rasa", RasaChatbot, description="RASA chatbot connector")
ChatbotFactory.register_chatbot("millionbot", MillionBot, description="MillionBot chatbot connector")

__all__ = [
    "Chatbot",
    "ChatbotConfig",
    "ChatbotFactory",
    "ChatbotResponse",
    "ChatbotTaskyto",
    "EndpointConfig",
    "Headers",
    "MillionBot",
    "Payload",
    "RasaChatbot",
    "RasaConfig",
    "RequestMethod",
    "ResponseProcessor",
    "SimpleTextProcessor",
    "TaskytoConfig",
]
