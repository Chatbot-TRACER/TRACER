"""Provides connector classes for interacting with different chatbot APIs."""

from typing import Any, TypedDict

import requests

# Define a common return type for execute_with_input
ChatbotResponse = tuple[bool, str | None]


class Chatbot:
    """Base class for chatbot connectors."""

    def __init__(self, url: str) -> None:
        """Initializes the base Chatbot connector.

        Args:
            url: The base URL for the chatbot API.
        """
        self.url: str = url
        self.fallback: str = "I do not understand you"  # Default fallback

    def execute_with_input(self, user_msg: str) -> ChatbotResponse:
        """Sends a message to the chatbot and returns the response.

        This method should be implemented by subclasses.

        Args:
            user_msg: The message from the user.

        Returns:
            A tuple containing:
                - bool: True if the interaction was successful, False otherwise.
                - str | None: The chatbot's response text, or an error message/None.
        """
        msg = "Subclasses must implement execute_with_input"
        raise NotImplementedError(msg)


class ChatbotTaskyto(Chatbot):
    """Connector for the Taskyto chatbot API."""

    def __init__(self, url: str) -> None:
        """Initializes the Taskyto connector."""
        super().__init__(url)
        self.id: str | None = None
        self.timeout: int = 20

    def _get_new_conversation_id(self) -> str | None:
        """Attempts to get a new conversation ID from the Taskyto API."""
        try:
            post_response = requests.post(self.url + "/conversation/new", timeout=self.timeout)
            post_response.raise_for_status()
            post_response_json = post_response.json()
            return post_response_json.get("id")
        except requests.exceptions.RequestException as e:
            print(f"Error getting new conversation ID: {e}")
        except requests.exceptions.JSONDecodeError:
            print("Error decoding JSON response for new conversation ID.")
        return None

    def execute_with_input(self, user_msg: str) -> ChatbotResponse:
        """Sends a message to the Taskyto chatbot and gets the response.

        Handles obtaining a conversation ID if one doesn't exist.

        Args:
            user_msg: The message from the user.

        Returns:
            A tuple containing:
                - bool: True if the interaction was successful, False otherwise.
                - str | None: The chatbot's response text, or an error message/None.
        """
        if self.id is None:
            self.id = self._get_new_conversation_id()
            if self.id is None:
                return False, "Failed to initialize conversation"

        new_data = {"id": self.id, "message": user_msg}
        success = False
        response_data: str | None = None

        try:
            post_response = requests.post(
                self.url + "/conversation/user_message",
                json=new_data,
                timeout=self.timeout,
            )
            post_response.raise_for_status()
            post_response_json = post_response.json()
            response_data = post_response_json.get("message")
            success = True

        except requests.Timeout:
            response_data = "timeout"
        except requests.exceptions.ConnectionError:
            response_data = "connection error"
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = http_err.response.json().get("error", str(http_err))
                response_data = f"HTTP error: {error_detail}"
            except requests.exceptions.JSONDecodeError:
                response_data = f"HTTP error: {http_err}"
        except requests.exceptions.JSONDecodeError:
            response_data = "invalid JSON response"
        except requests.exceptions.RequestException as req_err:
            response_data = f"request error: {req_err}"

        return success, response_data


class MillionBotConfig(TypedDict):
    """Configuration parameters for the MillionBot connector.

    Args:
        bot_id: The ID of the target bot.
        conversation_id: The ID for the conversation session.
        url: The URL associated with the chat context (e.g., webpage).
        sender: The sender ID.
        api_key: The API key for authorization.
        language: The language code (e.g., 'es').
    """

    bot_id: str
    conversation_id: str
    url: str
    sender: str
    api_key: str
    language: str


class MillionBot(Chatbot):
    """Connector for chatbots using the 1MillionBot API."""

    def __init__(self, url: str) -> None:
        """Initializes the MillionBot connector."""
        super().__init__(url)
        self.headers: dict[str, str] = {}
        self.payload: dict[str, Any] = {}
        self.id: str | None = None
        self.reset_url: str | None = None
        self.reset_payload: dict[str, Any] | None = None
        self.timeout: int = 10

    def init_chatbot(self, config: MillionBotConfig) -> None:
        """Configures the connector with specific MillionBot details.

        Args:
            config: A dictionary containing the necessary configuration parameters.
        """
        self.url = "https://api.1millionbot.com/api/public/messages"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"API-KEY {config['api_key']}",
        }
        self.payload = {
            "conversation": config["conversation_id"],
            "sender_type": "User",
            "sender": config["sender"],
            "bot": config["bot_id"],
            "language": config["language"],
            "url": config["url"],
            "message": {"text": "Hola"},
        }
        self.reset_url = "https://api.1millionbot.com/api/public/live/status"
        self.reset_payload = {
            "bot": config["bot_id"],
            "conversation": config["conversation_id"],
            "status": {
                "origin": config["url"],
                "online": False,
                "typing": False,
                "deleted": True,
                "attended": {},
                "userName": "ChatbotExplorer",  # Generic user name
            },
        }

    def _reset_conversation(self) -> bool:
        """Sends a reset request to the MillionBot API if needed."""
        if self.reset_url and self.reset_payload:
            try:
                response = requests.post(
                    self.reset_url, headers=self.headers, json=self.reset_payload, timeout=self.timeout
                )
                response.raise_for_status()
                self.reset_payload = None  # Clear after successful reset
            except requests.exceptions.RequestException as e:
                print(f"Error resetting conversation: {e}")
                return False
            else:
                return True
        return True  # No reset needed or possible

    def _process_response(self, response_json: dict[str, Any]) -> str:
        """Processes the JSON response from MillionBot to extract text and buttons."""
        text_response = ""
        for answer in response_json.get("response", []):
            if "text" in answer:
                text_response += answer["text"] + "\n"
            elif "payload" in answer:
                buttons_text = "\n\nAVAILABLE BUTTONS:\n\n"
                payload = answer["payload"]
                if "cards" in payload:
                    for card in payload.get("cards", []):
                        if "buttons" in card:
                            buttons_text += self._translate_buttons(card.get("buttons", []))
                elif "buttons" in payload:
                    buttons_text += self._translate_buttons(payload.get("buttons", []))
                # Only add button text if buttons were actually found
                if buttons_text.strip() != "AVAILABLE BUTTONS:":
                    text_response += buttons_text

        return text_response.strip()  # Remove trailing newline

    def execute_with_input(self, user_msg: str) -> ChatbotResponse:
        """Sends a message to the MillionBot chatbot and gets the response.

        Handles resetting the conversation state before the first message.

        Args:
            user_msg: The message from the user.

        Returns:
            A tuple containing:
                - bool: True if the interaction was successful, False otherwise.
                - str | None: The chatbot's response text, or an error message/None.
        """
        success = False
        response_data: str | None = None

        if not self._reset_conversation():
            print("Warning: Proceeding after failed conversation reset.")
            # Optionally return failure: return False, "Failed to reset conversation"

        self.payload["message"]["text"] = user_msg
        try:
            response = requests.post(self.url, headers=self.headers, json=self.payload, timeout=self.timeout)
            response.raise_for_status()
            response_json = response.json()
            response_data = self._process_response(response_json)
            success = True

        except requests.Timeout:
            response_data = "timeout"
        except requests.exceptions.ConnectionError:
            response_data = "connection error"
        except requests.exceptions.HTTPError as http_err:
            try:
                error_detail = http_err.response.json().get("error", str(http_err))
                print(f"Server error: {error_detail}")
                response_data = f"HTTP error: {error_detail}"
            except requests.exceptions.JSONDecodeError:
                print(f"Server error: {http_err}")
                response_data = f"HTTP error: {http_err}"
        except requests.exceptions.JSONDecodeError:
            response_data = "invalid JSON response"
        except requests.exceptions.RequestException as req_err:
            print(f"Request error: {req_err}")
            response_data = f"request error: {req_err}"

        return success, response_data

    def _translate_buttons(self, buttons_list: list[dict[str, Any]]) -> str:
        """Helper method to format button information into a string."""
        text_response = ""
        for button in buttons_list:
            button_text = button.get("text", "<No Text>")
            button_value = button.get("value", "<No Value>")  # Link or action value
            text_response += f"- BUTTON TEXT: {button_text} ACTION/LINK: {button_value}\n"
        return text_response


class ChatbotAdaUam(MillionBot):
    """Specific connector configuration for the ADA UAM chatbot."""

    def __init__(self, url: str | None = None) -> None:
        """Initializes the ADA UAM connector.

        Args:
            url: The base URL (optional, MillionBot overrides it).
        """
        super().__init__(url or "https://api.1millionbot.com")
        # Define configuration using the TypedDict
        config: MillionBotConfig = {
            "bot_id": "60a3be81f9a6b98f7659a6f9",
            "conversation_id": "670577afe0d59bbc894897b2",  # Consider making dynamic
            "url": "https://www.uam.es/uam/tecnologias-informacion",  # Context URL
            "sender": "670577af4e61b2bc9462703f",  # Consider making configurable
            "api_key": "60553d58c41f5dfa095b34b5",  # Consider making configurable
            "language": "es",
        }
        self.init_chatbot(config)
