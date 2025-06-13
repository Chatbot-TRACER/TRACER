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
ChatbotFactory.register_chatbot("taskyto", ChatbotTaskyto)
ChatbotFactory.register_chatbot("ada_uam", ChatbotAdaUam)
ChatbotFactory.register_chatbot("rasa", RasaChatbot)

__all__ = [
    "Chatbot",
    "ChatbotChatbotAdaUam",
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
