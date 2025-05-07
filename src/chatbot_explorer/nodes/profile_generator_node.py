"""LangGraph Node for Profile Generation."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.generation.profile_generation import (
    ProfileGenerationConfig,
    generate_profile_content,
)
from chatbot_explorer.schemas.graph_state_model import State
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_GOAL_PREVIEW_LENGTH = 70


def get_all_descriptions(nodes: list[dict[str, Any]]) -> list[str]:
    """Recursively extracts all 'description' values from a nested list of dictionaries.

    Args:
        nodes: A list of dictionaries, where each dictionary may contain a 'description' key
               and/or a 'children' key. The 'children' key, if present, contains another
               list of dictionaries with the same structure.

    Returns:
        A list of strings, where each string is a 'description' value found in the
        input list of dictionaries or any of its nested lists.
    """
    descriptions = []
    for node in nodes:
        if node.get("description"):
            descriptions.append(node["description"])
        if node.get("children"):
            child_descriptions = get_all_descriptions(node["children"])
            descriptions.extend(child_descriptions)
    return descriptions


def extract_function_details(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively extracts complete functionality information including parameters from nodes.

    Unlike get_all_descriptions which only extracts description strings, this function
    preserves the full functionality details including parameters.

    Args:
        nodes: A list of node dictionaries that may contain children

    Returns:
        A list of dictionaries containing name, description, and parameters for each node
    """
    details = []
    for node in nodes:
        if "name" in node and "description" in node:
            # Extract key functionality details
            func_detail = {
                "name": node["name"],
                "description": node["description"],
                "parameters": node.get("parameters", [])
            }
            details.append(func_detail)

        # Process children recursively
        if node.get("children"):
            child_details = extract_function_details(node["children"])
            details.extend(child_details)

    return details


def profile_generator_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Generates user profiles with conversation goals based on discovered functionalities.

    This function controls the generation of user profiles by using
    structured information about the chatbot's functionalities, limitations,
    and conversation history.

    Args:
        state (State): The current state of the chatbot exploration, containing
            information about discovered functionalities, limitations, conversation
            history, supported languages, and chatbot type.
        llm (BaseLanguageModel): The language model used for generating the
            user profiles.

    Returns:
        dict[str, Any]: A dictionary containing the generated user profiles
            under the key "conversation_goals". Returns an empty list if no
            functionalities are found, if an error occurs during profile
            generation, or if no descriptions are found in the structured
            functionalities.
    """
    if not state.get("discovered_functionalities"):
        logger.warning("Skipping goal generation: No structured functionalities found")
        return {"conversation_goals": []}

    # Functionalities are now dicts (structured from previous node)
    structured_root_dicts: list[dict[str, Any]] = state["discovered_functionalities"]

    # Get workflow structure (which is the structured functionalities itself)
    workflow_structure = structured_root_dicts  # Use the structured data directly

    # Get chatbot type from state
    chatbot_type = state.get("chatbot_type", "unknown")

    # Extract complete functionality details including parameters, not just descriptions
    functionality_full_details = extract_function_details(structured_root_dicts)

    # Still need descriptions for the profile generator
    functionality_descriptions = [func["description"] for func in functionality_full_details if "description" in func]

    if not functionality_descriptions:
        logger.warning("No descriptions found in structured functionalities")
        return {"conversation_goals": []}

    try:
        # Create the config dictionary
        config: ProfileGenerationConfig = {
            "functionalities": functionality_descriptions,
            "limitations": state.get("discovered_limitations", []),
            "llm": llm,
            "workflow_structure": workflow_structure,
            "conversation_history": state.get("conversation_history", []),
            "supported_languages": state.get("supported_languages", []),
            "chatbot_type": chatbot_type,
        }

        # Call the main generation function with the config dictionary
        profiles_with_goals = generate_profile_content(config)

    except (KeyError, TypeError, ValueError):
        logger.exception("Error during profile generation")
        return {"conversation_goals": []}  # Return empty list on error
    else:
        # Update state with the fully generated profiles
        return {"conversation_goals": profiles_with_goals}
