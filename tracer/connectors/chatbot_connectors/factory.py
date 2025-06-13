"""Factory for creating chatbot instances."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from tracer.connectors.chatbot_connectors.core import Chatbot


@dataclass
class ChatbotRegistration:
    """Registration metadata for a chatbot type."""

    chatbot_class: type
    factory_method: Callable[..., Chatbot]
    requires_url: bool = True
    description: str = ""


class ChatbotFactory:
    """Factory class for creating chatbot instances."""

    _chatbot_registrations: ClassVar[dict[str, ChatbotRegistration]] = {}

    @classmethod
    def register_chatbot(
        cls,
        name: str,
        chatbot_class: type,
        *,
        factory_method: Callable[..., Chatbot] | None = None,
        requires_url: bool = True,
        description: str = "",
    ) -> None:
        """Register a new chatbot type with its instantiation metadata.

        Args:
            name: Name identifier for the chatbot
            chatbot_class: The chatbot class
            factory_method: Custom factory method, defaults to direct instantiation
            requires_url: Whether this chatbot requires a URL parameter
            description: Description of the chatbot
        """
        if factory_method is None:
            factory_method = chatbot_class

        registration = ChatbotRegistration(
            chatbot_class=chatbot_class,
            factory_method=factory_method,
            requires_url=requires_url,
            description=description,
        )
        cls._chatbot_registrations[name] = registration

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of available chatbot types.

        Returns:
            List of registered chatbot type names
        """
        return list(cls._chatbot_registrations.keys())

    @classmethod
    def create_chatbot(cls, chatbot_type: str, base_url: str | None = None, **kwargs: str | int | bool) -> Chatbot:
        """Create a chatbot instance using registered factory method.

        Args:
            chatbot_type: Type of chatbot to create
            base_url: Base URL for the chatbot API (if required)
            **kwargs: Additional arguments to pass to the factory method

        Returns:
            Chatbot instance

        Raises:
            ValueError: If chatbot type is not registered or required URL is missing
        """
        if chatbot_type not in cls._chatbot_registrations:
            available = ", ".join(cls._chatbot_registrations.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)

        registration = cls._chatbot_registrations[chatbot_type]

        # Check if URL is required but not provided
        if registration.requires_url and base_url is None:
            error_msg = f"Chatbot type '{chatbot_type}' requires a base_url parameter"
            raise ValueError(error_msg)

        # Call the factory method with appropriate parameters
        try:
            if registration.requires_url:
                return registration.factory_method(base_url=base_url, **kwargs)
            return registration.factory_method(**kwargs)
        except TypeError as e:
            error_msg = f"Failed to create chatbot '{chatbot_type}': {e}"
            raise ValueError(error_msg) from e

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
        if chatbot_type not in cls._chatbot_registrations:
            available = ", ".join(cls._chatbot_registrations.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)

        return cls._chatbot_registrations[chatbot_type].chatbot_class

    @classmethod
    def requires_url(cls, chatbot_type: str) -> bool:
        """Check if a chatbot type requires a URL parameter.

        Args:
            chatbot_type: Type of chatbot to check

        Returns:
            True if the chatbot requires a URL parameter

        Raises:
            ValueError: If chatbot type is not registered
        """
        if chatbot_type not in cls._chatbot_registrations:
            available = ", ".join(cls._chatbot_registrations.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)

        return cls._chatbot_registrations[chatbot_type].requires_url
