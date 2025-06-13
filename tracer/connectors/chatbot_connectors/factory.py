"""Factory for creating chatbot instances."""

from typing import Any, ClassVar

from tracer.connectors.chatbot_connectors.core import Chatbot


class ChatbotFactory:
    """Factory class for creating chatbot instances."""

    _chatbot_classes: ClassVar[dict[str, type]] = {}

    @classmethod
    def register_chatbot(cls, name: str, chatbot_class: type) -> None:
        """Register a new chatbot type.

        Args:
            name: Name identifier for the chatbot
            chatbot_class: The chatbot class
        """
        cls._chatbot_classes[name] = chatbot_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available chatbot types.

        Returns:
            List of registered chatbot type names
        """
        return list(cls._chatbot_classes.keys())

    @classmethod
    def create_chatbot(cls, chatbot_type: str, **kwargs: dict[str, Any]) -> Chatbot:
        """Create a chatbot instance.

        Args:
            chatbot_type: Type of chatbot to create
            **kwargs: Arguments to pass to the chatbot constructor

        Returns:
            Chatbot instance

        Raises:
            ValueError: If chatbot type is not registered
        """
        if chatbot_type not in cls._chatbot_classes:
            available = ", ".join(cls._chatbot_classes.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)

        chatbot_class = cls._chatbot_classes[chatbot_type]
        return chatbot_class(**kwargs)

    @classmethod
    def get_chatbot_class(cls, chatbot_type: str) -> type:
        """Get the chatbot class for a given type.

        Args:
            chatbot_type: Type of chatbot to get class for

        Returns:
            The chatbot class

        Raises:
            ValueError: If chatbot type is not registered
        """
        if chatbot_type not in cls._chatbot_classes:
            available = ", ".join(cls._chatbot_classes.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)

        return cls._chatbot_classes[chatbot_type]
