"""RASA chatbot implementation."""

import uuid
from dataclasses import dataclass
from typing import Any

from tracer.connectors.chatbot_connectors.core import (
    Chatbot,
    ChatbotConfig,
    EndpointConfig,
    Payload,
    RequestMethod,
    ResponseProcessor,
)


class RasaResponseProcessor(ResponseProcessor):
    """Response processor for RASA chatbot."""

    def process(self, response_json: dict[str, Any]) -> str:
        """Process the RASA response JSON and extract messages.

        RASA returns a list of response objects, each potentially containing text,
        images, buttons, or other elements.

        Args:
            response_json: The JSON response from the RASA API (list of messages)

        Returns:
            Concatenated text from all response messages
        """
        if not isinstance(response_json, list):
            return ""

        text_parts = []
        for message in response_json:
            # Extract text content
            if "text" in message:
                text_parts.append(message["text"])

            # Handle buttons/quick replies
            if "buttons" in message:
                button_texts = [btn.get("title", btn.get("payload", "")) for btn in message["buttons"]]
                if button_texts:
                    text_parts.append(f"Options: {', '.join(button_texts)}")

            # Handle custom responses
            if "custom" in message:
                custom_text = str(message["custom"])
                text_parts.append(custom_text)

        return "\n".join(text_parts) if text_parts else ""


@dataclass
class RasaConfig(ChatbotConfig):
    """Configuration for RASA chatbot."""

    sender_id: str = "user"
    webhook_path: str = "/webhooks/rest/webhook"

    def __post_init__(self) -> None:
        """Set up headers after initialization."""
        self.headers = {"Content-Type": "application/json"}


class RasaChatbot(Chatbot):
    """Connector for RASA chatbot using REST webhook."""

    def __init__(self, base_url: str, sender_id: str = "user", timeout: int = 20) -> None:
        """Initialize the RASA chatbot connector.

        Args:
            base_url: The base URL for the RASA server
            sender_id: Unique identifier for the conversation sender
            timeout: Request timeout in seconds
        """
        config = RasaConfig(base_url=base_url, sender_id=sender_id, timeout=timeout)
        super().__init__(config)
        self.rasa_config = config

    def get_endpoints(self) -> dict[str, EndpointConfig]:
        """Return endpoint configurations for RASA chatbot."""
        return {
            "send_message": EndpointConfig(
                path=self.rasa_config.webhook_path, method=RequestMethod.POST, timeout=self.config.timeout
            )
        }

    def get_response_processor(self) -> ResponseProcessor:
        """Return the response processor for RASA chatbot."""
        return RasaResponseProcessor()

    def prepare_message_payload(self, user_msg: str) -> Payload:
        """Prepare the payload for sending a message to RASA.

        Args:
            user_msg: The user's message

        Returns:
            Payload dictionary for the RASA webhook request
        """
        return {"sender": self.rasa_config.sender_id, "message": user_msg}

    def _requires_conversation_id(self) -> bool:
        """RASA uses sender_id for conversation tracking."""
        return False

    def create_new_conversation(self) -> bool:
        """Create a new conversation by generating a new sender ID."""
        self.rasa_config.sender_id = f"user_{uuid.uuid4().hex[:8]}"
        return True
