import requests


class Chatbot:
    def __init__(self, url) -> None:
        self.url = url
        self.fallback = "I do not understand you"


class ChatbotTaskyto(Chatbot):
    def __init__(self, url) -> None:
        Chatbot.__init__(self, url)
        self.id = None

    def execute_with_input(self, user_msg):
        if self.id is None:
            try:
                post_response = requests.post(self.url + "/conversation/new")
                post_response_json = post_response.json()
                self.id = post_response_json.get("id")
            except requests.exceptions.ConnectionError:
                return False, "cut connection"
            except Exception:
                return False, "chatbot server error"

        if self.id is not None:
            new_data = {"id": self.id, "message": user_msg}

            try:
                timeout = 20
                try:
                    post_response = requests.post(
                        self.url + "/conversation/user_message",
                        json=new_data,
                        timeout=timeout,
                    )
                except requests.Timeout:
                    return False, "timeout"
                except requests.exceptions.ConnectionError:
                    return False, "chatbot internal error"

                post_response_json = post_response.json()

                if post_response.status_code == 200:
                    message = post_response_json.get("message")
                    # message = get_content(assistant_message) # get content process the message looking for images, pdf, or webpages
                    return True, message

                # There is an error, but it is an internal error
                return False, post_response_json.get("error")
            except requests.exceptions.JSONDecodeError:
                return False, "chatbot internal error"

        return True, ""


class MillionBot(Chatbot):
    def __init__(self, url) -> None:
        Chatbot.__init__(self, url)
        self.headers = {}
        self.payload = {}
        self.id = None

        self.reset_url = None
        self.reset_payload = None

    def init_chatbot(self, bot_id, conversation_id, url, sender="671ab2931382d56e5140f023") -> None:
        self.url = "https://api.1millionbot.com/api/public/messages"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "API-KEY 60553d58c41f5dfa095b34b5",
        }

        # Always generate a new ID for the conversation
        # import uuid
        # unique_id = uuid.uuid4()
        # conversation_id = unique_id.hex

        # Randomly replace a letter in the conversation_id with a hexadecimal digit
        # import random
        # import string
        # conversation_id = list(conversation_id)
        # conversation_id[random.randint(0, len(conversation_id)-1)] = random.choice(string.hexdigits)
        # conversation_id = ''.join(conversation_id)

        self.payload = {
            "conversation": conversation_id,
            "sender_type": "User",
            "sender": sender,
            "bot": bot_id,
            "language": "es",
            "url": url,
            "message": {"text": "Hola"},
        }

        self.reset_url = "https://api.1millionbot.com/api/public/live/status"
        self.reset_payload = {
            "bot": bot_id,
            "conversation": conversation_id,
            "status": {
                "origin": url,
                "online": False,
                "typing": False,
                "deleted": True,
                "attended": {},
                "userName": "UAM/UMU",
            },
        }
        self.timeout = 10

    def execute_with_input(self, user_msg):
        if self.reset_payload is not None:
            response = requests.post(self.reset_url, headers=self.headers, json=self.reset_payload)
            # print(response)
            assert response.status_code == 200
            self.reset_payload = None

        self.payload["message"]["text"] = user_msg
        timeout = self.timeout
        try:
            response = requests.post(self.url, headers=self.headers, json=self.payload, timeout=timeout)
            response_json = response.json()
            if response.status_code == 200:
                text_response = ""
                for answer in response_json["response"]:
                    # to-do --> pass the buttons in the answer?
                    if "text" in answer:
                        text_response += answer["text"] + "\n"
                    elif "payload" in answer:
                        text_response += "\n\nAVAILABLE BUTTONS:\n\n"
                        if "cards" in answer["payload"]:
                            for card in answer["payload"]["cards"]:
                                if "buttons" in card:
                                    text_response += self.__translate_buttons(card["buttons"])
                        elif "buttons" in answer["payload"]:
                            text_response += self.__translate_buttons(answer["payload"]["buttons"])

                return True, text_response
            # There is an error, but it is an internal error
            print(f"Server error {response_json.get('error')}")
            # errors.append({500: f"Couldn't get response from the server"})
            return False, response_json.get("error")
        except requests.exceptions.JSONDecodeError:
            # logger = logging.getLogger("my_app_logger")
            # logger.error(f"Couldn't get response from the server: {e}")
            return False, "chatbot internal error"
        except requests.Timeout:
            # logger = logging.getLogger("my_app_logger")
            # logger.error(
            #     f"No response was received from the server in less than {timeout}"
            # )
            # errors.append(
            #     {
            #         504: f"No response was received from the server in less than {timeout}"
            #     }
            # )
            return False, "timeout"

    def __translate_buttons(self, buttons_list) -> str:
        text_response = ""
        for button in buttons_list:
            if "text" in button:
                text_response += f"- BUTTON TEXT: {button['text']}"
            if "value" in button:
                text_response += f" LINK: {button['value']}\n"
            else:
                text_response += " LINK: <empty>\n"
        return text_response


class ChatbotAdaUam(MillionBot):
    def __init__(self, url) -> None:
        MillionBot.__init__(self, url)
        self.init_chatbot(
            bot_id="60a3be81f9a6b98f7659a6f9",
            conversation_id="670577afe0d59bbc894897b2",
            url="https://www.uam.es/uam/tecnologias-informacion",
            sender="670577af4e61b2bc9462703f",
        )
