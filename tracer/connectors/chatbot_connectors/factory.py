"""Factory for creating chatbot instances."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from tracer.connectors.chatbot_connectors.core import Chatbot, Parameter


@dataclass
class ChatbotRegistration:
    """Registration metadata for a chatbot type."""

    chatbot_class: type[Chatbot]
    factory_method: Callable[..., Chatbot]
    description: str = ""


class ChatbotFactory:
    """Factory class for creating chatbot instances."""

    _chatbot_registrations: ClassVar[dict[str, ChatbotRegistration]] = {}

    @classmethod
    def register_chatbot(
        cls,
        name: str,
        chatbot_class: type[Chatbot],
        *,
        factory_method: Callable[..., Chatbot] | None = None,
        description: str = "",
    ) -> None:
        """Register a new chatbot type with its instantiation metadata.

        Args:
            name: Name identifier for the chatbot
            chatbot_class: The chatbot class
            factory_method: Custom factory method, defaults to direct instantiation
            description: Description of the chatbot
        """
        if factory_method is None:
            factory_method = chatbot_class

        registration = ChatbotRegistration(
            chatbot_class=chatbot_class,
            factory_method=factory_method,
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
    def get_registered_connectors(cls) -> dict[str, dict[str, Any]]:
        """Get information about all registered chatbot connectors.

        Returns:
            Dictionary with connector names as keys and their metadata as values
        """
        return {
            name: {
                "description": registration.description,
                "chatbot_class": registration.chatbot_class.__name__,
            }
            for name, registration in cls._chatbot_registrations.items()
        }

    @classmethod
    def create_chatbot(cls, chatbot_type: str, **kwargs: Any) -> Chatbot:
        """Create a chatbot instance using registered factory method.

        Args:
            chatbot_type: Type of chatbot to create
            **kwargs: Arguments to pass to the factory method

        Returns:
            Chatbot instance

        Raises:
            ValueError: If chatbot type is not registered or required parameters are missing
        """
        registration = cls._get_registration(chatbot_type)
        cls._validate_parameters(chatbot_type, kwargs)

        try:
            return registration.factory_method(**kwargs)
        except TypeError as e:
            error_msg = f"Failed to create chatbot '{chatbot_type}': {e}"
            raise ValueError(error_msg) from e

    @classmethod
    def get_chatbot_parameters(cls, chatbot_type: str) -> list[Parameter]:
        """Get the parameters required for a given chatbot type.

        Args:
            chatbot_type: The type of chatbot

        Returns:
            A list of Parameter objects
        """
        registration = cls._get_registration(chatbot_type)
        return registration.chatbot_class.get_chatbot_parameters()

    @classmethod
    def _get_registration(cls, chatbot_type: str) -> ChatbotRegistration:
        """Get registration for a chatbot type, raising ValueError if not found."""
        if chatbot_type not in cls._chatbot_registrations:
            available = ", ".join(cls._chatbot_registrations.keys())
            error_msg = f"Unknown chatbot type: {chatbot_type}. Available: {available}"
            raise ValueError(error_msg)
        return cls._chatbot_registrations[chatbot_type]

    @classmethod
    def _validate_parameters(cls, chatbot_type: str, provided_args: dict[str, Any]) -> None:
        """Validate provided arguments against the chatbot's required parameters."""
        required_params = cls.get_chatbot_parameters(chatbot_type)

        missing_params = [
            param.name
            for param in required_params
            if param.required and param.name not in provided_args
        ]

        if missing_params:
            error_msg = f"Missing required parameters for '{chatbot_type}': {', '.join(missing_params)}"
            raise ValueError(error_msg)
