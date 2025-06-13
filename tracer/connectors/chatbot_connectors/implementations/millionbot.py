"""MillionBot chatbot implementations."""

from dataclasses import dataclass
from typing import Any

import requests

from tracer.connectors.chatbot_connectors.core import (
    Chatbot,
    ChatbotConfig,
    ChatbotResponse,
    EndpointConfig,
    Payload,
    RequestMethod,
    ResponseProcessor,
)
from tracer.utils.logging_utils import get_logger

logger = get_logger()


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
        return "\n".join(
            f"- BUTTON TEXT: {button.get('text', '<No Text>')} ACTION/LINK: {button.get('value', '<No Value>')}"
            for button in buttons_list
        )


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

    @classmethod
    def create_with_url(cls, base_url: str, bot_id: str, api_key: str, **config_kwargs: str | int) -> "MillionBot":
        """Factory method to create MillionBot instance with explicit parameters.

        Args:
            base_url: The base URL for the MillionBot API
            bot_id: The bot identifier
            api_key: The API key for authentication
            **config_kwargs: Additional configuration parameters

        Returns:
            MillionBot instance
        """
        config = MillionBotConfig(
            base_url=base_url,
            bot_id=bot_id,
            api_key=api_key,
            conversation_id=config_kwargs.get("conversation_id", ""),
            url_context=config_kwargs.get("url_context", ""),
            sender=config_kwargs.get("sender", ""),
            language=config_kwargs.get("language", "en"),
            timeout=int(config_kwargs.get("timeout", 20)),
        )
        return cls(config)

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
        except requests.RequestException:
            logger.exception("Error resetting conversation")
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
