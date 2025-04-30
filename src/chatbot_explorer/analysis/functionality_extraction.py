import re

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.conversation.conversation_utils import format_conversation
from chatbot_explorer.prompts.functionality_extraction_prompts import (
    get_functionality_extraction_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode


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
    # Format conversation for the LLM
    formatted_conversation = format_conversation(conversation_history)

    # Context for the LLM
    context = "Identify distinct interaction steps or functionalities the chatbot provides in this conversation, relevant to the user's workflow."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' step: {current_node.description}"

    extraction_prompt = get_functionality_extraction_prompt(
        context=context, formatted_conversation=formatted_conversation
    )

    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    print("\n--- Raw LLM Response for Functionality Extraction ---")
    print(content)
    print("-----------------------------------------------------")

    # Parse the LLM response
    functionality_nodes = []

    if "NO_NEW_FUNCTIONALITY" in content.upper():  # Case-insensitive check
        print("  LLM indicated no new functionalities.")
        return functionality_nodes

    # Split response into blocks
    blocks = re.split(r"FUNCTIONALITY:\s*", content, flags=re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block:  # Skip empty parts
            continue

        name = None
        description = None
        params_str = "None"

        # Parse lines in the block
        lines = block.split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("name:"):
                name = line[len("name:") :].strip()
            elif line.lower().startswith("description:"):
                description = line[len("description:") :].strip()
            elif line.lower().startswith("parameters:"):
                params_str = line[len("parameters:") :].strip()

        # Create node if we got name and description
        if name and description:
            # Parse parameters string
            parameters = []
            if params_str.lower() != "none":
                param_names = [p.strip() for p in params_str.split(",") if p.strip()]
                # Basic parameter structure
                parameters = [{"name": p, "type": "string", "description": f"Parameter {p}"} for p in param_names]

            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                parent=current_node,  # Set parent for now
            )
            functionality_nodes.append(new_node)
            print(f"  Identified step (Robust Parsing): {name}")
        elif block:  # Log blocks we couldn't parse
            print(f"  WARN: Could not parse functionality block:\n{block}")

    if not functionality_nodes and "NO_NEW_FUNCTIONALITY" not in content.upper():
        print("  WARN: LLM response did not contain 'NO_NEW_FUNCTIONALITY' but no functionalities were parsed.")

    return functionality_nodes
