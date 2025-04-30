"""LangGraph Node that builds the workflow structure from functionalities and conversation history."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.analysis.chatbot_classification import classify_chatbot_type
from chatbot_explorer.analysis.workflow_builder import build_workflow_structure
from chatbot_explorer.schemas.graph_state_model import State


def workflow_builder_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that analyzes functionalities and history to build the workflow structure.

    Uses different logic based on whether the bot seems transactional or informational.

    Args:
        state (State): The current graph state.
        llm: The language model instance.

    Returns:
        Dict[str, Any]: Dictionary with updated 'discovered_functionalities' and 'chatbot_type'.
    """
    print("\n--- Building Workflow Structure ---")
    # Functionalities are expected as dicts from run_full_exploration results
    flat_functionality_dicts = state.get("discovered_functionalities", [])
    conversation_history = state.get("conversation_history", [])

    if not flat_functionality_dicts:
        print("   Skipping structure building: No initial functionalities found.")
        # Return partial state update
        return {
            "discovered_functionalities": [],
            "chatbot_type": "unknown",  # Ensure type is set
        }

    # Classify the bot type first
    bot_type = classify_chatbot_type(flat_functionality_dicts, conversation_history, llm)
    print(f"   Chatbot type classified as: {bot_type}")

    try:
        structured_nodes = build_workflow_structure(flat_functionality_dicts, conversation_history, bot_type, llm)

        print(f"   Built structure with {len(structured_nodes)} root node(s).")

    except (ValueError, KeyError, TypeError) as e:
        # Handle errors during structure building
        print(f"   Error during structure building: {e}")
        # Keep the original flat list but update bot_type
        return {
            "discovered_functionalities": flat_functionality_dicts,
            "chatbot_type": bot_type,
        }
    else:
        # Update state with the final structured list of dictionaries and bot type
        return {
            "discovered_functionalities": structured_nodes,
            "chatbot_type": bot_type,
        }
