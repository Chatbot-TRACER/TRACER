"""Module to detect the supported language of the chatbot."""

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.language_detection_prompts import get_language_detection_prompt


def extract_supported_languages(chatbot_response: str, llm: BaseLanguageModel) -> list[str]:
    """Try to figure out what languages the chatbot knows.

    Args:
        chatbot_response (str): The chatbot's message.
        llm: The language model instance.

    Returns:
        List[str]: A list of language names (strings).
    """
    language_prompt = get_language_detection_prompt(chatbot_response)

    language_result = llm.invoke(language_prompt)
    languages = language_result.content.strip()

    # Clean up the LLM response
    languages = languages.replace("[", "").replace("]", "")
    return [lang.strip() for lang in languages.split(",") if lang.strip()]
