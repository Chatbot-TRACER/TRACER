"""Prompts for classifying chatbot interaction type (transactional vs. informational)."""


def get_classification_prompt(func_summary: str, conversation_snippets: str) -> str:
    """Generate the prompt for classifying the chatbot interaction type."""
    return f"""
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
