"""Module to build workflow structures from chatbot functionalities."""

import json
import pprint
import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.conversation.conversation_utils import format_conversation
from chatbot_explorer.prompts.workflow_prompts import (
    create_informational_prompt,
    create_transactional_prompt,
)
from chatbot_explorer.utils.logging_utils import get_logger
from chatbot_explorer.utils.parsing_utils import extract_json_from_response

logger = get_logger()


def _parse_and_validate_json_list(json_str: str) -> list[dict[str, Any]]:
    """Parses a JSON string and validates that it's a list of dictionaries.

    Args:
        json_str (str): The JSON string to parse.

    Returns:
        list[dict[str, Any]]: The parsed list of dictionaries.

    """
    parsed_data = json.loads(json_str)
    if not isinstance(parsed_data, list):
        msg = "LLM response is not a JSON list."
        raise TypeError(msg)
    return parsed_data


def build_node_hierarchy(structured_nodes_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a hierarchical structure from flat nodes with parent references.

    Args:
        structured_nodes_info: List of dictionaries containing node information.

    Returns:
        List[Dict[str, Any]]: List of root nodes with their children linked.

    """
    # Create a mapping of original node attributes by name for reference
    original_nodes_info = {node_info["name"]: node_info for node_info in structured_nodes_info if "name" in node_info}

    # Create working copy of nodes map to build hierarchy
    nodes_map = {node_info["name"]: node_info.copy() for node_info in structured_nodes_info if "name" in node_info}

    # Initialize children list for all nodes
    for node_info in nodes_map.values():
        node_info["children"] = []

    # Link children to parents based on 'parent_names'
    for node_name, node_info in nodes_map.items():
        parent_names = node_info.get("parent_names", [])
        for parent_name in parent_names:
            if parent_name in nodes_map:
                parent_node_info = nodes_map[parent_name]
                # Add child if not already present
                if node_info not in parent_node_info.get("children", []):
                    parent_node_info.setdefault("children", []).append(node_info)

    # Find root nodes (nodes that are not children of any other node)
    all_child_names = set()
    for node_info in nodes_map.values():
        for child_info in node_info.get("children", []):
            if isinstance(child_info, dict) and "name" in child_info:
                all_child_names.add(child_info["name"])

    root_nodes = [nodes_map[node_name] for node_name in nodes_map if node_name not in all_child_names]

    # Ensure all parameter and description information is preserved by checking original flat nodes
    # This handles cases where LLM might have omitted or simplified parameter data
    def restore_node_attributes(node):
        if "name" in node and node["name"] in original_nodes_info:
            original = original_nodes_info[node["name"]]

            # Preserve original parameters if they existed and are more detailed
            if "parameters" in original and (
                "parameters" not in node or
                not node["parameters"] or
                (isinstance(original["parameters"], list) and
                 len(original["parameters"]) > len(node.get("parameters", [])))
            ):
                node["parameters"] = original["parameters"]

            # Preserve original description if it was more detailed
            if "description" in original and len(original["description"]) > len(node.get("description", "")):
                node["description"] = original["description"]

            # Recursively process children
            for child in node.get("children", []):
                restore_node_attributes(child)

    # Apply restoration to all root nodes
    for root_node in root_nodes:
        restore_node_attributes(root_node)

    logger.debug("Node hierarchy built with %d root nodes from %d total nodes", len(root_nodes), len(nodes_map))
    return root_nodes


def build_workflow_structure(
    flat_functionality_dicts: list[dict[str, Any]],
    conversation_history: list[Any],
    chatbot_type: str,
    llm: BaseLanguageModel,
) -> list[dict[str, Any]]:
    """Build a hierarchical structure of chatbot functionalities.

    Args:
        flat_functionality_dicts: List of functionality dictionaries
        conversation_history: List of conversation sessions
        chatbot_type: Classification of the bot ("transactional" or "informational")
        llm: The language model instance

    Returns:
        List[Dict[str, Any]]: Structured hierarchy with parent-child relationships
    """
    logger.info("\n=== Building Workflow Structure ===\n")

    if not flat_functionality_dicts:
        logger.warning("Skipping structure building: No functionalities found")
        return []

    # Create a reference dictionary mapping functionality names to their full details
    # This will be used to restore parameter/description data that might be lost in LLM processing
    original_func_details = {}
    for func in flat_functionality_dicts:
        if "name" in func and func["name"]:
            original_func_details[func["name"]] = func
            # Log parameter counts for debugging
            param_count = len(func.get("parameters", []))
            if param_count > 0:
                logger.debug("Original functionality '%s' has %d parameters", func["name"], param_count)

    # Prepare functionality list string for prompt
    func_details_list = []
    func_list_str = "\n".join(
        [
            (
                f"- Name: {f.get('name', 'N/A')}\n"
                f"  Description: {f.get('description', 'N/A')}\n"
                f"  Parameters: [\n"
                + (
                    ",\n".join(
                        [
                            f"    {{'name': '{p.get('name', 'N/A')}', 'description': '{p.get('description', 'N/A')}', 'options': {p.get('options', [])}}}"
                            for p in f.get("parameters", [])
                        ]
                    )
                    or "    None"
                )
                + "\n  ]"
            )
            for f in flat_functionality_dicts
        ],
    )
    logger.debug("Prepared functionality list with %d entries", len(flat_functionality_dicts))

    # Extract conversation snippets
    snippets = []
    total_snippet_length = 0
    max_total_snippet_length = 7000  # Larger context for structure analysis

    if isinstance(conversation_history, list):
        for i, session_history in enumerate(conversation_history):
            if not isinstance(session_history, list):
                continue

            session_str = format_conversation(session_history)
            snippet_len = 1500

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

    conversation_snippets = "\n".join(snippets) or "No conversation history available."
    logger.debug("Prepared %d conversation snippets (%d total characters)", len(snippets), total_snippet_length)

    # Select appropriate prompt based on chatbot type
    if chatbot_type == "transactional":
        logger.debug("Using transactional structuring prompt for chatbot workflow")
        structuring_prompt = create_transactional_prompt(func_list_str, conversation_snippets)
    else:  # Default to informational
        logger.debug("Using informational structuring prompt for chatbot workflow")
        structuring_prompt = create_informational_prompt(func_list_str, conversation_snippets)

    try:
        logger.debug("Invoking LLM to generate workflow structure")
        response = llm.invoke(structuring_prompt)
        response_content = response.content

        json_str = extract_json_from_response(response_content)

        # Clean up potential JSON issues
        json_str = re.sub(r"//.*?(\n|$)", "\n", json_str)  # Remove comments
        json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)  # Remove trailing commas

        structured_nodes_info = _parse_and_validate_json_list(json_str)
        logger.debug("Successfully parsed workflow structure JSON with %d nodes", len(structured_nodes_info))

        # Restore original parameter data from flat functionalities where appropriate
        for node in structured_nodes_info:
            if "name" in node and node["name"] in original_func_details:
                original = original_func_details[node["name"]]

                # Check parameters field and restore if original has more complete data
                if "parameters" in original and isinstance(original["parameters"], list):
                    # If LLM provided no parameters or fewer parameters than original, restore original
                    if not node.get("parameters") or not isinstance(node["parameters"], list) or len(node["parameters"]) < len(original["parameters"]):
                        logger.debug(
                            "Restoring original %d parameters for '%s' (LLM provided %s)",
                            len(original["parameters"]),
                            node["name"],
                            len(node.get("parameters", [])) if isinstance(node.get("parameters"), list) else "none"
                        )
                        node["parameters"] = original["parameters"]

        # Log the structured_nodes_info (LLM output before hierarchy building)
        logger.debug(
            "  LLM Output (structured_nodes_info):\n%s", pprint.pformat(structured_nodes_info, indent=2, width=120)
        )

        root_nodes_dicts = build_node_hierarchy(structured_nodes_info)
        logger.debug(
            "Final hierarchical workflow structure built (%d root nodes). Checking for parameter options:",
            len(root_nodes_dicts),
        )
        logger.debug(
            "  Hierarchical Workflow Structure (root_nodes_dicts):\n%s",
            pprint.pformat(root_nodes_dicts, indent=2, width=120),
        )

    except (json.JSONDecodeError, TypeError):
        logger.exception("Failed to parse or validate JSON from LLM response")
        logger.verbose("Returning original flat functionality list as fallback")
        return flat_functionality_dicts  # Return original list on failure
    else:
        return root_nodes_dicts
