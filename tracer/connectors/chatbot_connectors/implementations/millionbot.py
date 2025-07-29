"""MillionBot chatbot implementation."""

from dataclasses import dataclass
from typing import Any

import requests

from tracer.connectors.chatbot_connectors.core import (
    Chatbot,
    ChatbotConfig,
    EndpointConfig,
    Parameter,
    Payload,
    RequestMethod,
    ResponseProcessor,
)


class MillionBotResponseProcessor(ResponseProcessor):
    """Response processor for MillionBot chatbot."""
    def process(self, response_json: dict[str, Any]) -> str:
        """Process the MillionBot response JSON and extract messages."""
        text_response = ""
        for answer in response_json.get("response", []):
            if "text" in answer:
                text_response += answer["text"] + "\n"
            elif "payload" in answer:
                text_response += "\n\nAVAILABLE BUTTONS:\n\n"
                if "cards" in answer["payload"]:
                    for card in answer["payload"]["cards"]:
                        if "buttons" in card:
                            text_response += self._translate_buttons(card["buttons"])
                elif "buttons" in answer["payload"]:
                    text_response += self._translate_buttons(
                        answer["payload"]["buttons"]
                    )
        return text_response

    def _translate_buttons(self, buttons_list: list[dict[str, Any]]) -> str:
        """Translate a list of buttons to a string."""
        text_response = ""
        for button in buttons_list:
            if "text" in button:
                text_response += f"- BUTTON TEXT: {button['text']}"
            if "value" in button:
                text_response += f" LINK: {button['value']}\n"
            else:
                text_response += " LINK: <empty>\n"
        return text_response


@dataclass
class MillionBotConfig(ChatbotConfig):
    """Configuration for the MillionBot chatbot."""

    bot_id: str = ""


class MillionBot(Chatbot):
    """Connector for the MillionBot chatbot API."""

    def __init__(self, bot_id: str, timeout: int = 20) -> None:
        """Initialize the MillionBot chatbot connector."""
        config = MillionBotConfig(
            base_url="https://api.1millionbot.com/api/public/", bot_id=bot_id, timeout=timeout
        )
        super().__init__(config)
        self.millionbot_config = config
        self._initialize_conversation()

    @classmethod
    def get_chatbot_parameters(cls) -> list[Parameter]:
        """Return the parameters required to initialize this chatbot."""
        return [
            Parameter(
                name="bot_id",
                type="string",
                required=True,
                description="The Bot ID for the MillionBot.",
            )
        ]

    def _initialize_conversation(self) -> None:
        """Initialize the conversation with the MillionBot API."""
        # Step 1: Create user - using minimal required fields based on API example
        user_payload = {
            "bot": self.millionbot_config.bot_id,
            "language": "es-ES",
            "platform": "Win32",
            "country": "Spain",
            "countryData": {"isoCode": "ES", "name": "Spain"},
            "timezone": "Europe/Madrid",
            "ip": "127.0.0.1"  # Using localhost IP as default
        }
        user_headers = {
            "Content-Type": "application/json",
            "Authorization": "API-KEY 60553d58c41f5dfa095b34b5",
        }
        user_url = self.config.get_full_url("users")
        user_response = self.session.post(
            user_url, headers=user_headers, json=user_payload, timeout=self.config.timeout
        )
        user_response.raise_for_status()
        user_data = user_response.json()
        user_id = user_data["user"]["_id"]

        # Step 2: Create conversation
        conversation_payload = {
            "bot": self.millionbot_config.bot_id,
            "user": user_id,
            "language": "es",
            "integration": "web",
            "gdpr": True
        }
        conversation_headers = {
            "Content-Type": "application/json",
            "Authorization": "60a3bee2e3987316fed3218f",
        }
        conversation_url = self.config.get_full_url("conversations")
        conversation_response = self.session.post(
            conversation_url,
            headers=conversation_headers,
            json=conversation_payload,
            timeout=self.config.timeout,
        )
        conversation_response.raise_for_status()
        conversation_data = conversation_response.json()
        self.conversation_id = conversation_data["conversation"]["_id"]
        self.user_id = user_id
        self.session.headers.update(conversation_headers)

    def get_endpoints(self) -> dict[str, EndpointConfig]:
        """Return endpoint configurations for MillionBot chatbot."""
        return {
            "send_message": EndpointConfig(
                path="/messages", method=RequestMethod.POST, timeout=self.config.timeout
            )
        }

    def get_response_processor(self) -> ResponseProcessor:
        """Return the response processor for MillionBot chatbot."""
        return MillionBotResponseProcessor()

    def prepare_message_payload(self, user_msg: str) -> Payload:
        """Prepare the payload for sending a message to MillionBot."""
        payload = {
            "conversation": self.conversation_id,
            "sender_type": "User",
            "sender": self.user_id,
            "bot": self.millionbot_config.bot_id,
            "language": "es",
            "message": {"text": user_msg},
        }
        return payload

    def _requires_conversation_id(self) -> bool:
        return True

    def create_new_conversation(self) -> bool:
        """Create a new conversation for MillionBot."""
        try:
            self._initialize_conversation()
            return True
        except Exception as e:
            return False
