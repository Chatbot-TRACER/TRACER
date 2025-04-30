import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.conversation.conversation_utils import format_conversation
from chatbot_explorer.prompts.functionality_extraction_prompts import (
    get_functionality_extraction_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode


def _parse_parameter_string(params_str: str) -> list[dict[str, Any]]:
    """Parses the 'parameters' string into a list of parameter dictionaries."""
    parameters: list[dict[str, Any]] = []
    if params_str.lower().strip() != "none":
        param_names = [p.strip() for p in params_str.split(",") if p.strip()]
        # Basic parameter structure assumption
        parameters = [{"name": p, "type": "string", "description": f"Parameter {p}"} for p in param_names]
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
        print("  LLM indicated no new functionalities.")
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
            parameters = _parse_parameter_string(params_str)

            # Create the new node
            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                parent=current_node,  # Assign parent based on context
            )
            functionality_nodes.append(new_node)
            print(f"  Identified step: {name}")
        # Log blocks that couldn't be fully parsed but were not empty
        elif block_content:
            print(f"  WARN: Could not parse functionality block:\n{block_content}")

    # Final check if parsing yielded nothing despite no explicit 'NO_NEW' flag
    if not functionality_nodes and "NO_NEW_FUNCTIONALITY" not in content.upper():
        print("  WARN: LLM response did not contain 'NO_NEW_FUNCTIONALITY' but no functionalities were parsed.")

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
    # 1. Format conversation for the LLM
    formatted_conversation = format_conversation(conversation_history)

    # 2. Prepare context for the LLM prompt
    context = "Identify distinct interaction steps or functionalities the chatbot provides in this conversation, relevant to the user's workflow."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' step: {current_node.description}"

    # 3. Get the prompt and invoke the LLM
    extraction_prompt = get_functionality_extraction_prompt(
        context=context, formatted_conversation=formatted_conversation
    )
    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    print("\n--- Raw LLM Response for Functionality Extraction ---")
    print(content)
    print("-----------------------------------------------------")

    # 4. Parse the LLM response using the helper function
    return _parse_llm_functionality_response(content, current_node)
