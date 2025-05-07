"""Module to extract functionalities (as Functionality Nodes) from conversations."""

import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.conversation.conversation_utils import format_conversation
from chatbot_explorer.prompts.functionality_extraction_prompts import (
    get_functionality_extraction_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode, ParameterDefinition
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

CONTENT_PREVIEW_LENGTH = 100

# Patterns for identifying meta-functionalities
META_FUNCTIONALITY_PATTERNS = [
    r"(ask|explain|list|describe|provide|tell)_(capabilities|what_i_can_do|features|help|services)",
    r"(introduce|present|state)_(self|capabilities|purpose|services|functions)",
    r"(ask|request|inquire)_how_to_help",
    r"(greet|welcome)_user",
    r"user_(asks|requests|inquires)_(about|for)_(capabilities|help|information)",
    r"(offer|provide)_(assistance|help)",
    r"explain_(purpose|what_i_do|how_i_work)",
    r"(general|initial)_(greeting|introduction)",
]

def _is_meta_functionality(node_name: str, node_description: str) -> bool:
    """Determines if a node represents a meta-functionality that should be filtered out.

    Args:
        node_name: The name of the functionality node
        node_description: The description of the functionality node

    Returns:
        True if the node is a meta-functionality, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    name_lower = node_name.lower()
    desc_lower = node_description.lower()

    # Check name against patterns
    for pattern in META_FUNCTIONALITY_PATTERNS:
        if re.search(pattern, name_lower):
            logger.debug("Filtered meta-functionality by name pattern: '%s'", node_name)
            return True

    # Check for specific meta-functionality words in name
    meta_words = ["capability", "capabilities", "help", "assistance", "introduction", "greeting"]
    for word in meta_words:
        if word in name_lower:
            logger.debug("Filtered meta-functionality by name keyword: '%s'", node_name)
            return True

    # Check description for meta-functionality indicators
    meta_phrases = [
        "what the chatbot can do",
        "chatbot's capabilities",
        "chatbot introduces itself",
        "chatbot asks how it can help",
        "chatbot greets the user",
        "user asks what the chatbot can do",
        "explains its capabilities",
        "introduces itself",
        "welcomes the user"
    ]

    for phrase in meta_phrases:
        if phrase in desc_lower:
            logger.debug("Filtered meta-functionality by description phrase: '%s'", node_name)
            return True

    return False

def _filter_meta_functionalities(nodes: list[FunctionalityNode]) -> list[FunctionalityNode]:
    """Filters out nodes that represent meta-functionalities.

    Args:
        nodes: List of functionality nodes to filter

    Returns:
        Filtered list of nodes with meta-functionalities removed
    """
    initial_count = len(nodes)
    filtered_nodes = [node for node in nodes if not _is_meta_functionality(node.name, node.description)]

    filtered_count = initial_count - len(filtered_nodes)
    if filtered_count > 0:
        logger.debug("Filtered out %d meta-functionality nodes", filtered_count)

    return filtered_nodes


def _parse_parameter_string(params_str: str) -> list[dict[str, Any]]:
    """Parses the 'parameters' string into a list of parameter dictionaries with options.

    Args:
        params_str: String containing parameters, possibly with options in parentheses
                   Format: "param1 (option1/option2): description, param2: description, param3 (optA/optB): description"

    Returns:
        List of parameter dictionaries with name, type, description, and options
    """
    parameters: list[ParameterDefinition] = []
    if params_str.lower().strip() == "none":
        return parameters

    # Split parameters and extract options and descriptions
    param_parts = params_str.split(",")

    for param_part in param_parts:
        param_part = param_part.strip()
        if not param_part:
            continue

        # Extract description if present
        param_name_and_desc = param_part.split(":", 1)
        param_name_with_options = param_name_and_desc[0].strip()
        description = param_name_and_desc[1].strip() if len(param_name_and_desc) > 1 else f"Parameter {param_name_with_options.split()[0]}"

        # Extract options if present
        options_match = re.search(r"\s*\(([^)]+)\)", param_name_with_options)
        options = []
        if options_match:
            options_str = options_match.group(1)
            options = [opt.strip() for opt in options_str.split("/") if opt.strip()]
            # Remove options part from param_name
            param_name = re.sub(r"\s*\([^)]+\)", "", param_name_with_options).strip()
        else:
            param_name = param_name_with_options.strip()

        # Create a ParameterDefinition object
        param = ParameterDefinition(
            name=param_name,
            description=description,
            options=options
        )
        parameters.append(param)

    return parameters


def _parse_single_functionality_block(block: str) -> tuple[str, str, str] | None:
    """Parses a single block of text for functionality details."""
    name = None
    description = None
    params_str = "None"

    # Parse lines within the block to find name, description, parameters
    lines = block.split("\n")
    for line in lines:
        line_content = line.strip()
        # Extract name if found
        if line_content.lower().startswith("name:"):
            name = line_content[len("name:") :].strip()
        # Extract description if found
        elif line_content.lower().startswith("description:"):
            description = line_content[len("description:") :].strip()
        # Extract parameters string if found
        elif line_content.lower().startswith("parameters:"):
            params_str = line_content[len("parameters:") :].strip()

    # Return parsed details only if essential information is present
    if name and description:
        return name, description, params_str
    # Return None if parsing failed for this block
    return None


def _parse_llm_functionality_response(content: str, current_node: FunctionalityNode | None) -> list[FunctionalityNode]:
    """Parses the raw LLM response string to extract FunctionalityNode objects."""
    functionality_nodes = []

    # Check if the LLM explicitly stated no new functionality
    if "NO_NEW_FUNCTIONALITY" in content.upper():
        logger.debug("LLM indicated no new functionalities")
        return functionality_nodes

    # Split response into potential functionality blocks
    blocks = re.split(r"FUNCTIONALITY:\s*", content, flags=re.IGNORECASE)

    # Process each block
    for block in blocks:
        block_content = block.strip()
        # Skip empty parts resulting from the split
        if not block_content:
            continue

        # Attempt to parse the block using the helper function
        parsed_details = _parse_single_functionality_block(block_content)

        # If parsing was successful, create a node
        if parsed_details:
            name, description, params_str = parsed_details
            # Parse the parameters string using its helper function
            parameter_dicts = _parse_parameter_string(params_str)

            # Convert parameter dictionaries to ParameterDefinition objects
            parameters = []
            if isinstance(parameter_dicts, list):
                parameters = parameter_dicts
            else:
                logger.warning("Expected a list of parameters but got: %s", type(parameter_dicts))

            # Create the new node
            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                parent=current_node,  # Assign parent based on context
            )
            functionality_nodes.append(new_node)
            logger.debug("Identified functionality: '%s'", name)
        # Log blocks that couldn't be fully parsed but were not empty
        elif block_content:
            logger.warning(
                "Could not parse functionality block: %s",
                block_content[:CONTENT_PREVIEW_LENGTH] + ("..." if len(block_content) > CONTENT_PREVIEW_LENGTH else ""),
            )

    # Final check if parsing yielded nothing despite no explicit 'NO_NEW' flag
    if not functionality_nodes and "NO_NEW_FUNCTIONALITY" not in content.upper():
        logger.warning("LLM response did not contain 'NO_NEW_FUNCTIONALITY' but no functionalities were parsed")

    return functionality_nodes


def extract_functionality_nodes(
    conversation_history: list,
    llm: BaseLanguageModel,
    current_node: FunctionalityNode | None = None,
) -> list[FunctionalityNode]:
    """Find out FunctionalityNodes from the conversation.

    Args:
        conversation_history (list): The list of chat messages.
        llm: The language model instance.
        current_node (FunctionalityNode, optional): The node being explored. Defaults to None.

    Returns:
        List[FunctionalityNode]: A list of newly found FunctionalityNode objects.
    """
    logger.verbose("Extracting functionality nodes from conversation")

    # 1. Format conversation for the LLM
    formatted_conversation = format_conversation(conversation_history)
    logger.debug("Formatted conversation for functionality extraction")

    # 2. Prepare context for the LLM prompt
    context = "Identify distinct interaction steps or functionalities the chatbot provides in this conversation, relevant to the user's workflow."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' step: {current_node.description}"
        logger.debug("Extraction context includes current node: '%s'", current_node.name)

    # 3. Get the prompt and invoke the LLM
    extraction_prompt = get_functionality_extraction_prompt(
        context=context, formatted_conversation=formatted_conversation
    )
    logger.debug("Invoking LLM for functionality extraction")
    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    logger.debug("\n--- Raw LLM Response for Functionality Extraction ---")
    # Split by lines to make it more readable in logs
    for line in content.split("\n"):
        if line.strip():
            logger.debug("%s", line)
    logger.debug("-----------------------------------------------------")

    # 4. Parse the LLM response using the helper function
    nodes = _parse_llm_functionality_response(content, current_node)

    # 5. Filter out meta-functionalities
    nodes = _filter_meta_functionalities(nodes)

    logger.debug("Extracted %d functionality nodes after filtering", len(nodes))

    return nodes
