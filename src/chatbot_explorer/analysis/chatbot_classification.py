from typing import Any

from chatbot_explorer.conversation.conversation_utils import format_conversation


def classify_chatbot_type(functionalities: list[dict[str, Any]], conversation_history: list[Any], llm) -> str:
    """Determine if the chatbot is transactional (task-oriented) or informational (Q&A).

    Args:
        functionalities: List of functionality dictionaries
        conversation_history: List of conversation sessions
        llm: The language model instance

    Returns:
        str: "transactional", "informational", or "unknown"
    """
    print("\n--- Classifying Chatbot Interaction Type ---")

    if not conversation_history or not functionalities:
        print("   Skipping classification: Insufficient data.")
        return "unknown"

    # Create a summary of functionality names and descriptions
    func_summary = "\n".join(
        [
            f"- {f.get('name', 'N/A')}: {f.get('description', 'N/A')[:100]}..."
            for f in functionalities[:10]  # Limit to first 10 for summary
        ],
    )

    # Get conversation snippets
    snippets = []
    total_snippet_length = 0
    max_total_snippet_length = 5000  # Limit context size

    if isinstance(conversation_history, list):
        for i, session_history in enumerate(conversation_history):
            if not isinstance(session_history, list):
                continue

            session_str = format_conversation(session_history)
            snippet_len = 1000  # Max length per snippet

            # Take beginning and end if too long
            session_snippet = (
                session_str[: snippet_len // 2] + "\n...\n" + session_str[-snippet_len // 2 :]
                if len(session_str) > snippet_len
                else session_str
            )

            # Add snippet if within total length limit
            if total_snippet_length + len(session_snippet) < max_total_snippet_length:
                snippets.append(f"\n--- Snippet from Session {i + 1} ---\n{session_snippet}")
                total_snippet_length += len(session_snippet)
            else:
                break  # Stop if limit reached

    conversation_snippets = "\n".join(snippets) or "No conversation snippets available."

    # Create prompt for classification
    classification_prompt = f"""
    Analyze the following conversation snippets and discovered functionalities to classify the chatbot's primary interaction style.

    Discovered Functionalities Summary:
    {func_summary}

    Conversation Snippets:
    {conversation_snippets}

    Consider these definitions:
    - **Transactional / Workflow-driven:** The chatbot guides the user through a specific multi-step process with clear sequences, choices, and goals (e.g., ordering food, booking an appointment, completing a form). Conversations often involve the chatbot asking questions to gather input and presenting options to advance the workflow.
    - **Informational / Q&A:** The chatbot primarily answers user questions on various independent topics. Users typically ask a question, get an answer (often text or links), and might then ask about a completely different topic. There isn't usually a strict required sequence between topics.

    Based on the overall pattern in the conversations and the nature of the functionalities, is this chatbot PRIMARILY Transactional/Workflow-driven or Informational/Q&A?

    Respond with ONLY ONE word: "transactional" or "informational".
    """

    try:
        # Ask the LLM for classification
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()

        if classification in ["transactional", "informational"]:
            print(f"   LLM classified as: {classification}")
            return classification
        # Handle unclear response
        print(f"   LLM response unclear ('{classification}'), defaulting to informational.")
        return "informational"
    except Exception as e:
        # Handle LLM error
        print(f"   Error during classification: {e}. Defaulting to informational.")
        return "informational"
