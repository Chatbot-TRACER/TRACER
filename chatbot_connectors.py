import requests


class Chatbot:
    def __init__(self, url):
        self.url = url
        self.fallback = "I do not understand you"


class ChatbotTaskyto(Chatbot):
    def __init__(self, url):
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
                except requests.exceptions.ConnectionError as e:
                    return False, "chatbot internal error"

                post_response_json = post_response.json()

                if post_response.status_code == 200:
                    message = post_response_json.get("message")
                    # message = get_content(assistant_message) # get content process the message looking for images, pdf, or webpages
                    return True, message

                else:
                    # There is an error, but it is an internal error
                    return False, post_response_json.get("error")
            except requests.exceptions.JSONDecodeError as e:
                return False, "chatbot internal error"

        return True, ""

    def execute_starter_chatbot(self):
        timeout = 20
        try:
            post_response = requests.post(self.url + "/conversation/new")
            post_response_json = post_response.json()
            self.id = post_response_json.get("id")
            if post_response.status_code == 200:
                message = post_response_json.get("message")
                # message = get_content(assistant_message)
                if message is None:
                    return True, "Hello"
                else:
                    return True, message
            else:
                # There is an error, but it is an internal error
                return False, post_response_json.get("error")
        except requests.exceptions.ConnectionError:
            return False, "cut connection"
        except requests.Timeout:
            return False, "timeout"
