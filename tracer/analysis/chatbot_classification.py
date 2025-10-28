"""Classify chatbots as transactional or informational based on their functionalities and conversation history."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.conversation.conversation_utils import format_conversation
from tracer.prompts.classification_prompts import get_classification_prompt
from tracer.utils.logging_utils import get_logger

logger = get_logger()


def classify_chatbot_type(
    functionalities: list[dict[str, Any]],
    conversation_history: list[Any],
    llm: BaseLanguageModel,
) -> str:
    """Determine if the chatbot is transactional (task-oriented) or informational (Q&A).

    Args:
        functionalities: List of functionality dictionaries
        conversation_history: List of conversation sessions
        llm: The language model instance

    Returns:
        str: "transactional", "informational", or "unknown"
    """
    if not conversation_history or not functionalities:
        logger.warning("Skipping classification: Insufficient data")
        return "unknown"

    # Create a summary of functionality names and descriptions
    func_summary = "\n".join(
        [
            f"- {f.get('name', 'N/A')}: {f.get('description', 'N/A')[:100]}..."
            for f in functionalities[:10]  # Limit to first 10 for summary
        ],
    )

    logger.debug("Prepared functionality summary with %d entries", min(len(functionalities), 10))

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
    logger.debug("Prepared %d conversation snippets (%d total characters)", len(snippets), total_snippet_length)

    classification_prompt = get_classification_prompt(
        func_summary=func_summary,
        conversation_snippets=conversation_snippets,
    )

    try:
        # Ask the LLM for classification
        logger.debug("Invoking LLM for chatbot classification")
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()

        if classification in ["transactional", "informational"]:
            logger.debug("LLM classified chatbot as: %s", classification)
            return classification

        # Handle unclear response
        logger.warning("LLM classification response unclear ('%s'), defaulting to informational", classification)

    except (ValueError, TypeError):
        # Handle LLM error with logger.exception
        logger.exception("Error during chatbot classification. Defaulting to informational")
        return "informational"
    else:
        return "informational"
