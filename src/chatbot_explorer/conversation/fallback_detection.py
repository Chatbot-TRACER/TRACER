"""Module to detect and analyze fallback messages from the chatbot."""

import re

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.fallback_detection_prompts import (
    get_fallback_identification_prompt,
    get_semantic_fallback_check_prompt,
)
from connectors.chatbot_connectors import Chatbot


def extract_fallback_message(the_chatbot: Chatbot, llm: BaseLanguageModel) -> str | None:
    """Try to get the chatbot's fallback message.

    Sends confusing messages to trigger it. These aren't part of the main chat history.

    Args:
        the_chatbot: The chatbot connector instance.
        llm: The language model instance.

    Returns:
        Optional[str]: The detected fallback message, or None if not found.
    """
    print("\n--- Attempting to detect chatbot fallback message (won't be included in analysis) ---")

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
        print(f"\nSending confusing query {i + 1}...")
        try:
            is_ok, response = the_chatbot.execute_with_input(query)

            if is_ok:
                print(f"Response received ({len(response)} chars)")
                responses.append(response)
        except (TimeoutError, ConnectionError) as e:
            print(f"Error communicating with chatbot: {e}")

    # Analyze responses if we got any
    if responses:
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

            fallback_preview_length = 50
            if fallback:
                print(
                    f'Detected fallback pattern: "{fallback[:fallback_preview_length]}{"..." if len(fallback) > fallback_preview_length else ""}"'
                )
                return fallback
            print("Could not extract a clear fallback message pattern.")
        except (TimeoutError, ConnectionError, ValueError) as e:
            print(f"Error during fallback analysis: {e}")

    print("Could not detect a consistent fallback message.")
    return None


def is_semantically_fallback(response: str, fallback: str, llm: BaseLanguageModel) -> bool:
    """Check if the chatbot's response is semantically equivalent to a known fallback message.

    Args:
        response (str): The chatbot's current response.
        fallback (str): The known fallback message pattern.
        llm: The language model instance.

    Returns:
        bool: True if the response is considered a fallback, False otherwise.
    """
    if not response or not fallback:
        return False  # Cannot compare if one is empty

    prompt = get_semantic_fallback_check_prompt(response, fallback)

    try:
        llm_decision = llm.invoke(prompt)
        decision_text = llm_decision.content.strip().upper()

        return decision_text.startswith("YES")
    except (TimeoutError, ConnectionError, ValueError) as e:
        print(f"   LLM Fallback Check Error: {e}. Assuming not a fallback.")
        return False  # Default to False if LLM fails
