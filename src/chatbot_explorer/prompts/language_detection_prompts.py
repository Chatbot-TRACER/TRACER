"""Prompts for detecting languages supported by a chatbot."""


def get_language_detection_prompt(chatbot_response: str) -> str:
    """Generate a prompt for detecting the languages supported by a chatbot."""
    return f"""
    Based on the following chatbot response, determine what language(s) the chatbot supports.
    If the response is in a non-English language, include that language in the list.
    If the response explicitly mentions supported languages, list those.

    CHATBOT RESPONSE:
    {chatbot_response}

    FORMAT YOUR RESPONSE AS A COMMA-SEPARATED LIST OF LANGUAGES:
    [language1, language2, ...]

    RESPONSE:
    """
