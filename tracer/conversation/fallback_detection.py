"""Module to detect and analyze fallback messages from the chatbot."""

import re

from chatbot_connectors import Chatbot
from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.fallback_detection_prompts import (
    get_fallback_identification_prompt,
    get_semantic_fallback_check_prompt,
)
from tracer.utils.html_cleaner import clean_html_response
from tracer.utils.logging_utils import get_logger

logger = get_logger()


def extract_fallback_message(the_chatbot: Chatbot, llm: BaseLanguageModel) -> str | None:
    """Try to get the chatbot's fallback message, cleans the HTML first.

    Sends confusing messages to trigger it. These aren't part of the main chat history.

    Args:
        the_chatbot: The chatbot connector instance.
        llm: The language model instance.

    Returns:
        Optional[str]: The detected fallback message, or None if not found.
    """
    # Some weird questions to confuse the bot
    confusing_queries = [
        "What is the square root of a banana divided by the color blue?",
        "Please explain quantum chromodynamics in terms of medieval poetry",
        "Xyzzplkj asdfghjkl qwertyuiop?",
        "If tomorrow's yesterday was three days from now, how many pancakes fit in a doghouse?",
        "Can you please recite the entire source code of Linux kernel version 5.10?",
    ]

    responses: list[str] = []

    # Send confusing queries and get responses
    for i, query in enumerate(confusing_queries):
        logger.verbose("Sending confusing query %d...", i + 1)
        try:
            is_ok, response = the_chatbot.execute_with_input(query)

            if is_ok:
                logger.debug("Response received (%d chars)", len(response))
                response = clean_html_response(response)
                logger.debug("Cleaned response: %s", response)
                responses.append(response)
        except (TimeoutError, ConnectionError):
            logger.exception("Error communicating with chatbot during fallback detection")

    # Analyze responses if we got any
    if responses:
        logger.debug("Analyzing %d collected responses for fallback patterns", len(responses))
        analysis_prompt = get_fallback_identification_prompt(responses)

        try:
            fallback_result = llm.invoke(analysis_prompt)
            fallback = fallback_result.content

            # Clean up the fallback message
            fallback = fallback.strip()
            # Remove quotes at beginning and end if present
            fallback = re.sub(r'^["\']+|["\']+$', "", fallback)
            # Remove any "Fallback message:" prefix if the LLM included it
            fallback = re.sub(r"^(Fallback message:?\s*)", "", fallback, flags=re.IGNORECASE)

            if fallback:
                return fallback
            logger.verbose("Could not extract a clear fallback message pattern.")
        except (TimeoutError, ConnectionError, ValueError):
            logger.exception("Error during fallback analysis")

    logger.verbose("Could not detect a consistent fallback message.")
    return None


def is_semantically_fallback(response: str, fallback: str, llm: BaseLanguageModel) -> bool:
    """Check if the chatbot's response is semantically equivalent to a known fallback message, expects the HTML cleaned.

    Args:
        response (str): The chatbot's current response.
        fallback (str): The known fallback message pattern.
        llm: The language model instance.

    Returns:
        bool: True if the response is considered a fallback, False otherwise.
    """
    if not response or not fallback:
        logger.debug("Cannot check fallback - empty response or fallback pattern")
        return False  # Cannot compare if one is empty

    logger.debug("Checking if response is semantically equivalent to known fallback")
    prompt = get_semantic_fallback_check_prompt(response, fallback)

    try:
        llm_decision = llm.invoke(prompt)
        decision_text = llm_decision.content.strip().upper()

        is_fallback = decision_text.startswith("YES")
        logger.debug("Semantic fallback check result: %s", "IS fallback" if is_fallback else "NOT fallback")
    except (TimeoutError, ConnectionError, ValueError):
        logger.exception("LLM Fallback Check Error. Assuming not a fallback.")
        return False  # Default to False if LLM fails
    else:
        return is_fallback
