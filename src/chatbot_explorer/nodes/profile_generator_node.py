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
    logger.info("Generating profiles for %s chatbot\n", chatbot_type)

    functionality_descriptions = get_all_descriptions(structured_root_dicts)

    if not functionality_descriptions:
        logger.warning("No descriptions found in structured functionalities")
        return {"conversation_goals": []}

    logger.info("Processing %d functional components for profile generation\n", len(functionality_descriptions))

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

        # Log the results with profile names
        profile_count = len(profiles_with_goals)
        logger.info("Generated %d user profiles:", profile_count)
        for i, profile in enumerate(profiles_with_goals, 1):
            name = profile.get("name", f"Profile {i}")
            logger.info(" • %s", name)

            # Log variable count at verbose level
            variables = [k for k, v in profile.items() if isinstance(v, dict) and "function" in v and "data" in v]
            if variables:
                logger.verbose("   Variables: %d (%s)", len(variables), ", ".join(variables))

    except (KeyError, TypeError, ValueError):
        logger.exception("Error during profile generation")
        return {"conversation_goals": []}  # Return empty list on error
    else:
        # Update state with the fully generated profiles
        return {"conversation_goals": profiles_with_goals}
