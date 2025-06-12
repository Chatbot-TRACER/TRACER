"""Provides an extensible framework for chatbot API connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
from urllib.parse import urljoin

import requests

# Type aliases
ChatbotResponse = tuple[bool, str | None]
Headers = dict[str, str]
Payload = dict[str, Any]


class RequestMethod(Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class EndpointConfig:
    """Configuration for API endpoints."""

    path: str
    method: RequestMethod = RequestMethod.POST
    headers: Headers = field(default_factory=dict)
    timeout: int = 20


@dataclass
class ChatbotConfig:
    """Base configuration for chatbot connectors."""

    base_url: str
    timeout: int = 20
    fallback_message: str = "I do not understand you"
    headers: Headers = field(default_factory=dict)

    def get_full_url(self, endpoint: str) -> str:
        """Construct full URL from base URL and endpoint."""
        return urljoin(self.base_url, endpoint.lstrip("/"))


class ResponseProcessor(ABC):
    """Abstract base class for processing chatbot responses."""

    @abstractmethod
    def process(self, response_json: dict[str, Any]) -> str:
        """Process the JSON response and extract meaningful text.

        Args:
            response_json: The JSON response from the API

        Returns:
            Processed response text
        """


class SimpleTextProcessor(ResponseProcessor):
    """Simple processor that extracts text from a specified field."""

    def __init__(self, text_field: str = "message") -> None:
        """Initialize the processor with the field to extract text from.

        Args:
            text_field: The field name to extract text from in the response JSON.
        """
        self.text_field = text_field

    def process(self, response_json: dict[str, Any]) -> str:
        """Extract text from the specified field in the response JSON.

        Args:
            response_json: The JSON response from the API.

        Returns:
            Extracted text from the specified field, or an empty string if not found.
        """
        return response_json.get(self.text_field, "")


class Chatbot(ABC):
    """Abstract base class for chatbot connectors with common functionality."""

    def __init__(self, config: ChatbotConfig) -> None:
        """Initialize the chatbot connector.

        Args:
            config: The configuration for the chatbot connector.
        """
        self.config = config
        self.session = requests.Session()
        self.conversation_id: str | None = None
        self._setup_session()

    def _setup_session(self) -> None:
        """Set up the requests session with default headers."""
        self.session.headers.update(self.config.headers)

    @abstractmethod
    def get_endpoints(self) -> dict[str, EndpointConfig]:
        """Return endpoint configurations for this chatbot.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """

    @abstractmethod
    def get_response_processor(self) -> ResponseProcessor:
        """Return the response processor for this chatbot."""

    @abstractmethod
    def prepare_message_payload(self, user_msg: str) -> Payload:
        """Prepare the payload for sending a message.

        Args:
            user_msg: The user's message

        Returns:
            Payload dictionary for the API request
        """

    def create_new_conversation(self) -> bool:
        """Create a new conversation.

        Default implementation that can be overridden by subclasses.

        Returns:
            True if successful, False otherwise
        """
        endpoints = self.get_endpoints()
        if "new_conversation" not in endpoints:
            # If no new conversation endpoint, just reset the conversation ID
            self.conversation_id = None
            return True

        endpoint_config = endpoints["new_conversation"]
        url = self.config.get_full_url(endpoint_config.path)

        try:
            response = self._make_request(url, endpoint_config, {})
            if response:
                # Try to extract conversation ID if provided
                self.conversation_id = response.get("id") or response.get("conversation_id")
                return True
        except requests.RequestException as e:
            print(f"Error creating new conversation: {e}")

        return False

    def execute_with_input(self, user_msg: str) -> ChatbotResponse:
        """Send a message to the chatbot and get the response.

        Args:
            user_msg: The user's message

        Returns:
            Tuple of (success, response_text)
        """
        # Ensure we have a conversation if needed
        if self.conversation_id is None and self._requires_conversation_id() and not self.create_new_conversation():
            return False, "Failed to initialize conversation"

        endpoints = self.get_endpoints()
        if "send_message" not in endpoints:
            return False, "Send message endpoint not configured"

        endpoint_config = endpoints["send_message"]
        url = self.config.get_full_url(endpoint_config.path)
        payload = self.prepare_message_payload(user_msg)

        try:
            response_json = self._make_request(url, endpoint_config, payload)
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError, requests.RequestException) as e:
            error_map = {
                requests.Timeout: "timeout",
                requests.ConnectionError: "connection error",
                requests.HTTPError: f"HTTP error: {e}",
                requests.RequestException: f"request error: {e}",
            }
            error_message = error_map.get(type(e), f"request error: {e}")
            return False, error_message

        if response_json:
            processor = self.get_response_processor()
            response_text = processor.process(response_json)
            return True, response_text

        return False, "No response received"

    def _requires_conversation_id(self) -> bool:
        """Check if this chatbot requires a conversation ID.

        Can be overridden by subclasses.
        """
        return True

    def _make_request(self, url: str, endpoint_config: EndpointConfig, payload: Payload) -> dict[str, Any] | None:
        """Make an HTTP request with error handling.

        Args:
            url: The request URL
            endpoint_config: Endpoint configuration
            payload: Request payload

        Returns:
            JSON response or None if failed
        """
        headers = {**self.session.headers, **endpoint_config.headers}

        if endpoint_config.method == RequestMethod.GET:
            response = self.session.get(url, params=payload, headers=headers, timeout=endpoint_config.timeout)
        else:
            response = self.session.request(
                endpoint_config.method.value, url, json=payload, headers=headers, timeout=endpoint_config.timeout
            )

        response.raise_for_status()
        return response.json()


# Specific implementations


class TaskytoResponseProcessor(ResponseProcessor):
    """Response processor for Taskyto chatbot."""

    def process(self, response_json: dict[str, Any]) -> str:
        """Extract the message from the Taskyto response JSON.

        Args:
            response_json: The JSON response from the API.

        Returns:
            The message string from the response.
        """
        return response_json.get("message", "")


class TaskytoConfig(ChatbotConfig):
    """Configuration specific to Taskyto chatbot."""


class ChatbotTaskyto(Chatbot):
    """Connector for the Taskyto chatbot API."""

    def __init__(self, base_url: str, timeout: int = 20) -> None:
        """Initialize the Taskyto chatbot connector.

        Args:
            base_url: The base URL for the Taskyto API.
            timeout: Request timeout in seconds.
        """
        config = TaskytoConfig(base_url=base_url, timeout=timeout)
        super().__init__(config)

    def get_endpoints(self) -> dict[str, EndpointConfig]:
        """Return endpoint configurations for Taskyto chatbot."""
        return {
            "new_conversation": EndpointConfig(
                path="/conversation/new", method=RequestMethod.POST, timeout=self.config.timeout
            ),
            "send_message": EndpointConfig(
                path="/conversation/user_message", method=RequestMethod.POST, timeout=self.config.timeout
            ),
        }

    def get_response_processor(self) -> ResponseProcessor:
        """Return the response processor for Taskyto chatbot."""
        return TaskytoResponseProcessor()

    def prepare_message_payload(self, user_msg: str) -> Payload:
        """Prepare the payload for sending a message to Taskyto.

        Args:
            user_msg: The user's message.

        Returns:
            Payload dictionary for the API request.
        """
        return {"id": self.conversation_id, "message": user_msg}


class MillionBotResponseProcessor(ResponseProcessor):
    """Response processor for MillionBot API."""

    def process(self, response_json: dict[str, Any]) -> str:
        """Process the MillionBot response JSON and extract text and buttons.

        Args:
            response_json: The JSON response from the API.

        Returns:
            The processed response text including available buttons if present.
        """
        text_response = ""
        for answer in response_json.get("response", []):
            if "text" in answer:
                text_response += answer["text"] + "\n"
            elif "payload" in answer:
                buttons_text = self._process_buttons(answer["payload"])
                if buttons_text:
                    text_response += f"\n\nAVAILABLE BUTTONS:\n\n{buttons_text}"

        return text_response.strip()

    def _process_buttons(self, payload: dict[str, Any]) -> str:
        """Process buttons from payload."""
        buttons_text = ""

        # Handle cards with buttons
        if "cards" in payload:
            for card in payload.get("cards", []):
                if "buttons" in card:
                    buttons_text += self._format_buttons(card.get("buttons", []))

        # Handle direct buttons
        elif "buttons" in payload:
            buttons_text += self._format_buttons(payload.get("buttons", []))

        return buttons_text

    def _format_buttons(self, buttons_list: list) -> str:
        """Format button list into text."""
        text = ""
        for button in buttons_list:
            button_text = button.get("text", "<No Text>")
            button_value = button.get("value", "<No Value>")
            text += f"- BUTTON TEXT: {button_text} ACTION/LINK: {button_value}\n"
        return text


@dataclass
class MillionBotConfig(ChatbotConfig):
    """Configuration for MillionBot chatbots."""

    bot_id: str = ""
    conversation_id: str = ""
    url_context: str = ""
    sender: str = ""
    api_key: str = ""
    language: str = "en"

    def __post_init__(self) -> None:
        """Set up headers with API key after initialization."""
        self.headers = {"Content-Type": "application/json", "Authorization": f"API-KEY {self.api_key}"}


class MillionBot(Chatbot):
    """Connector for chatbots using the 1MillionBot API."""

    def __init__(self, config: MillionBotConfig) -> None:
        """Initialize the MillionBot connector.

        Args:
            config: The configuration for the MillionBot chatbot.
        """
        super().__init__(config)
        self.mb_config = config
        self.reset_needed = True

    def get_endpoints(self) -> dict[str, EndpointConfig]:
        """Return endpoint configurations for MillionBot chatbot."""
        return {
            "send_message": EndpointConfig(
                path="/api/public/messages", method=RequestMethod.POST, timeout=self.config.timeout
            ),
            "reset_conversation": EndpointConfig(
                path="/api/public/live/status", method=RequestMethod.POST, timeout=self.config.timeout
            ),
        }

    def get_response_processor(self) -> ResponseProcessor:
        """Return the response processor for MillionBot chatbot."""
        return MillionBotResponseProcessor()

    def prepare_message_payload(self, user_msg: str) -> Payload:
        """Prepare the payload for sending a message to MillionBot.

        Args:
            user_msg: The user's message.

        Returns:
            Payload dictionary for the API request.
        """
        return {
            "conversation": self.mb_config.conversation_id,
            "sender_type": "User",
            "sender": self.mb_config.sender,
            "bot": self.mb_config.bot_id,
            "language": self.mb_config.language,
            "url": self.mb_config.url_context,
            "message": {"text": user_msg},
        }

    def _requires_conversation_id(self) -> bool:
        return False  # MillionBot uses config-based conversation ID

    def create_new_conversation(self) -> bool:
        """Reset conversation state."""
        self.reset_needed = True
        return True

    def execute_with_input(self, user_msg: str) -> ChatbotResponse:
        """Send message with conversation reset if needed."""
        if self.reset_needed and not self._reset_conversation():
            return False, "Failed to reset conversation"

        return super().execute_with_input(user_msg)

    def _reset_conversation(self) -> bool:
        """Reset the conversation state."""
        endpoints = self.get_endpoints()
        if "reset_conversation" not in endpoints:
            return True

        endpoint_config = endpoints["reset_conversation"]
        url = self.config.get_full_url(endpoint_config.path)

        reset_payload = {
            "bot": self.mb_config.bot_id,
            "conversation": self.mb_config.conversation_id,
            "status": {
                "origin": self.mb_config.url_context,
                "online": False,
                "typing": False,
                "deleted": True,
                "attended": {},
                "userName": "ChatbotExplorer",
            },
        }

        try:
            self._make_request(url, endpoint_config, reset_payload)
            self.reset_needed = False
        except requests.RequestException as e:
            print(f"Error resetting conversation: {e}")
            return False
        else:
            return True


class ChatbotAdaUam(MillionBot):
    """Pre-configured connector for the ADA UAM chatbot."""

    def __init__(self) -> None:
        """Initialize the ADA UAM chatbot connector."""
        config = MillionBotConfig(
            base_url="https://api.1millionbot.com",
            bot_id="60a3be81f9a6b98f7659a6f9",
            conversation_id="670577afe0d59bbc894897b2",
            url_context="https://www.uam.es/uam/tecnologias-informacion",
            sender="670577af4e61b2bc9462703f",
            api_key="60553d58c41f5dfa095b34b5",
            language="es",
        )
        super().__init__(config)


# Factory for easy chatbot creation
class ChatbotFactory:
    """Factory class for creating chatbot instances."""

    _chatbot_classes: ClassVar[dict[str, type]] = {
        "taskyto": ChatbotTaskyto,
        "millionbot": MillionBot,
        "ada_uam": ChatbotAdaUam,
    }

    @classmethod
    def register_chatbot(cls, name: str, chatbot_class: type) -> None:
        """Register a new chatbot type.

        Args:
            name: Name identifier for the chatbot
            chatbot_class: The chatbot class
        """
        cls._chatbot_classes[name] = chatbot_class

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
