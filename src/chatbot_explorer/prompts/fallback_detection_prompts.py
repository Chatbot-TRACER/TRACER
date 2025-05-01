"""Prompts for detecting and analyzing fallback messages from the chatbot."""


def get_fallback_identification_prompt(responses: list[str]) -> str:
    """Generate the prompt for identifying the chatbot's fallback message pattern."""
    return f"""
    I'm trying to identify a chatbot's fallback message - the standard response it gives when it doesn't understand.

    Below are responses to intentionally confusing or nonsensical questions.
    If there's a consistent pattern or identical response, that's likely the fallback message.

    RESPONSES:
    {responses}

    ANALYSIS STEPS:
    1. Check for identical responses - if any responses are exactly the same, that's likely the fallback.
    2. Look for very similar responses with only minor variations.
    3. Identify common phrases or sentence patterns across responses.

    EXTRACT ONLY THE MOST LIKELY FALLBACK MESSAGE OR PATTERN.
    If the fallback message appears to have minor variations, extract the common core part that appears in most responses.
    Do not include any analysis, explanation, or quotation marks in your response.
    """


def get_semantic_fallback_check_prompt(response: str, fallback: str) -> str:
    """Generate the prompt for semantically comparing a response to a known fallback pattern."""
    return f"""
    Compare the following two messages. Determine if the "Chatbot Response" is semantically equivalent to the "Known Fallback Message".

    "Semantically equivalent" means the response conveys the same core meaning as the fallback, such as:
    - Not understanding the request.
    - Being unable to process the request.
    - Asking the user to rephrase.
    - Stating a general limitation.

    It does NOT have to be an exact word-for-word match.

    Known Fallback Message:
    "{fallback}"

    Chatbot Response:
    "{response}"

    Is the "Chatbot Response" semantically equivalent to the "Known Fallback Message"?

    Respond with ONLY "YES" or "NO".
    """
