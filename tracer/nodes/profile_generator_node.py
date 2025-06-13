"""LangGraph Node for Profile Generation."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.generation.profile_generation import (
    ProfileGenerationConfig,
    generate_profile_content,
)
from tracer.schemas.graph_state_model import State
from tracer.utils.logging_utils import get_logger

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


def _extract_parameter_info(parameters: list[dict[str, Any]]) -> list[str]:
    """Extract parameter information from parameter list.

    Args:
        parameters: List of parameter dictionaries.

    Returns:
        List of formatted parameter strings.
    """
    param_info = []
    for param in parameters:
        if not isinstance(param, dict):
            continue

        param_name = param.get("name", "")
        if not param_name:
            continue

        param_text = param_name
        if param_desc := param.get("description", ""):
            param_text += f": {param_desc}"
        if param_options := param.get("options", []):
            param_text += f" [options: {', '.join(param_options)}]"
        param_info.append(param_text)

    return param_info


def _extract_output_info(outputs: list[dict[str, Any]]) -> list[str]:
    """Extract output information from output list.

    Args:
        outputs: List of output dictionaries.

    Returns:
        List of formatted output strings.
    """
    output_info = []
    for output in outputs:
        if not isinstance(output, dict):
            continue

        category = output.get("category", "")
        description = output.get("description", "")
        if category and description:
            output_info.append(f"{category}: {description}")

    return output_info


def _build_functionality_text(node: dict[str, Any]) -> str:
    """Build functionality text with parameter and output information.

    Args:
        node: Node dictionary containing functionality information.

    Returns:
        Formatted functionality text string.
    """
    functionality_text = node["description"]

    # Add parameter information
    parameters = node.get("parameters", [])
    if parameters and isinstance(parameters, list):
        param_info = _extract_parameter_info(parameters)
        if param_info:
            functionality_text += f" (Inputs: {'; '.join(param_info)})"

    # Add output information
    outputs = node.get("outputs", [])
    if outputs and isinstance(outputs, list):
        output_info = _extract_output_info(outputs)
        if output_info:
            functionality_text += f" (Outputs: {'; '.join(output_info)})"

    return functionality_text


def get_functionalities_with_outputs(nodes: list[dict[str, Any]]) -> list[str]:
    """Recursively extracts descriptions and output options from functionality nodes.

    Args:
        nodes: A list of dictionaries, where each dictionary may contain 'description',
              'outputs', and 'children' keys.

    Returns:
        A list of strings, where each string combines the functionality description
        with its output information if available.
    """
    functionality_info = []

    for node in nodes:
        if node.get("description"):
            functionality_text = _build_functionality_text(node)
            functionality_info.append(functionality_text)

        # Recursively process children
        if node.get("children"):
            child_info = get_functionalities_with_outputs(node["children"])
            functionality_info.extend(child_info)

    return functionality_info


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

    # Get functionalities with their outputs
    functionalities_with_outputs = get_functionalities_with_outputs(structured_root_dicts)

    if not functionalities_with_outputs:
        logger.warning("No functionalities with outputs found in structured functionalities")
        return {"conversation_goals": []}

    logger.debug(
        ">>> Rich Functionality Strings Prepared for Profile Grouping (%d total):", len(functionalities_with_outputs)
    )
    for i, rich_string in enumerate(functionalities_with_outputs):
        logger.debug("  RICH_FUNC_STRING[%d]: %s", i, rich_string)
    logger.debug("<<< End of Rich Functionality Strings")

    # Get nested_forward parameter from state
    nested_forward = state.get("nested_forward", False)
    if nested_forward:
        logger.info("Nested forward chaining is enabled for variable definitions")

    try:
        # Create the config dictionary
        config: ProfileGenerationConfig = {
            "functionalities": functionalities_with_outputs,
            "limitations": state.get("discovered_limitations", []),
            "llm": llm,
            "workflow_structure": workflow_structure,
            "conversation_history": state.get("conversation_history", []),
            "supported_languages": state.get("supported_languages", []),
            "chatbot_type": chatbot_type,
        }

        # Call the main generation function with the config dictionary
        profiles_with_goals = generate_profile_content(config, nested_forward=nested_forward)

    except (KeyError, TypeError, ValueError):
        logger.exception("Error during profile generation")
        return {"conversation_goals": []}  # Return empty list on error
    else:
        # Update state with the fully generated profiles
        return {"conversation_goals": profiles_with_goals}
