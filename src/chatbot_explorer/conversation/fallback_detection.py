import re


def extract_fallback_message(the_chatbot, llm) -> str | None:
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

    responses = []

    # Send confusing queries and get responses
    for i, query in enumerate(confusing_queries):
        print(f"\nSending confusing query {i + 1}...")
        try:
            is_ok, response = the_chatbot.execute_with_input(query)

            if is_ok:
                print(f"Response received ({len(response)} chars)")
                responses.append(response)
        except Exception as e:
            print(f"Error communicating with chatbot: {e}")

    # Analyze responses if we got any
    if responses:
        analysis_prompt = f"""
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
                print(f'Detected fallback pattern: "{fallback[:50]}{"..." if len(fallback) > 50 else ""}"')
                return fallback
            print("Could not extract a clear fallback message pattern.")
        except Exception as e:
            print(f"Error during fallback analysis: {e}")

    print("Could not detect a consistent fallback message.")
    return None


def is_semantically_fallback(response: str, fallback: str, llm) -> bool:
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

    prompt = f"""
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
    try:
        llm_decision = llm.invoke(prompt)
        decision_text = llm_decision.content.strip().upper()

        return decision_text.startswith("YES")
    except Exception as e:
        print(f"   LLM Fallback Check Error: {e}. Assuming not a fallback.")
        return False  # Default to False if LLM fails
