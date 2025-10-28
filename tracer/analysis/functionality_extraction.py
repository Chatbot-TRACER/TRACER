"""Module to extract functionalities (as Functionality Nodes) from conversations."""

import re

from langchain_core.language_models import BaseLanguageModel

from tracer.conversation.conversation_utils import format_conversation
from tracer.prompts.functionality_extraction_prompts import (
    get_functionality_extraction_prompt,
)
from tracer.schemas.functionality_node_model import FunctionalityNode, OutputOptions, ParameterDefinition
from tracer.utils.logging_utils import get_logger

logger = get_logger()

CONTENT_PREVIEW_LENGTH = 100


def _parse_parameter_string(params_str: str) -> list[ParameterDefinition]:
    """Parses the 'parameters' string into a list of parameter dictionaries with options.

    Args:
        params_str: String containing parameters, possibly with options in parentheses and descriptions after colon
                   Format: "param1 (option1/option2): description1, param2: description2"

    Returns:
        List of ParameterDefinition objects
    """
    parameters: list[ParameterDefinition] = []
    if params_str.lower().strip() == "none":
        return parameters

    # Split the parameters by comma, but not inside parentheses
    param_entries = []
    start = 0
    paren_level = 0
    for i, char in enumerate(params_str):
        if char == "(":
            paren_level += 1
        elif char == ")":
            paren_level -= 1
        elif char == "," and paren_level == 0:
            param_entries.append(params_str[start:i].strip())
            start = i + 1
    # Add the last parameter
    if start < len(params_str):
        param_entries.append(params_str[start:].strip())

    for param_entry in param_entries:
        # Extract parameter name and options (if any)
        param_info, *description_parts = param_entry.split(":", 1)
        param_description = description_parts[0].strip() if description_parts else ""

        # Extract name and options
        if "(" in param_info and ")" in param_info:
            parts = param_info.split("(", 1)
            param_name = parts[0].strip()
            options_str = parts[1].split(")", 1)[0].strip()
            options = [opt.strip() for opt in options_str.split("/") if opt.strip()]
        else:
            param_name = param_info.strip()
            options = []

        if not param_name:
            continue

        # Create a ParameterDefinition object with a meaningful description
        param = ParameterDefinition(
            name=param_name,
            description=param_description
            if param_description
            else f"Specifies the {param_name} value for this functionality",
            options=options,
        )
        parameters.append(param)

    return parameters


def _parse_output_options_string(output_str: str) -> list[OutputOptions]:
    """Parses the 'output_options' string into a list of OutputOptions objects.

    Args:
        output_str: String containing output options by category
                    Format: "category1: description1; category2: description2"

    Returns:
        List of OutputOptions objects
    """
    output_options: list[OutputOptions] = []
    if output_str.lower().strip() == "none":
        return output_options

    # Split by category (each separated by semicolon)
    category_pattern = re.compile(r"([^;:]+):\s*([^;]+)(?:;|$)")
    category_matches = category_pattern.findall(output_str)

    for category_name, description in category_matches:
        category = category_name.strip()
        if not category:
            continue

        # Create an OutputOptions object
        output_option = OutputOptions(category=category, description=description.strip())
        output_options.append(output_option)

    return output_options


def _parse_single_functionality_block(block: str) -> tuple[str, str, str, str] | None:
    """Parses a single block of text for functionality details."""
    name = None
    description = None
    params_str = "None"
    output_str = "None"

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
        # Extract output options string if found
        elif line_content.lower().startswith("output_options:"):
            output_str = line_content[len("output_options:") :].strip()

    # Return parsed details only if essential information is present
    if name and description:
        return name, description, params_str, output_str
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
            name, description, params_str, output_str = parsed_details
            # Parse the parameters string using its helper function
            parameters = _parse_parameter_string(params_str)

            # Parse the output options string using its helper function
            output_options = _parse_output_options_string(output_str)

            # Filter out any None/null values in the output_options list
            output_options = [opt for opt in output_options if opt is not None]

            # Ensure we never create a node with [null] outputs
            if not output_options and output_str.strip().lower() != "none":
                logger.warning("Failed to parse any output options from: '%s'", output_str)
                # Set to empty list explicitly for clarity
                output_options = []

            # Create the new node
            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                outputs=output_options,
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

    logger.debug("--- Raw LLM Response for Functionality Extraction ---")
    # Split by lines to make it more readable in logs
    for line in content.split("\n"):
        if line.strip():
            logger.debug("%s", line)
    logger.debug("-----------------------------------------------------")

    # 4. Parse the LLM response using the helper function
    nodes = _parse_llm_functionality_response(content, current_node)
    logger.debug("Extracted %d functionality nodes", len(nodes))

    return nodes
