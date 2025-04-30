import json
import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.workflow_prompts import (
    create_informational_prompt,
    create_transactional_prompt,
)
from chatbot_explorer.utils.parsing_utils import extract_json_from_response


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
    nodes_map = {node_info["name"]: node_info for node_info in structured_nodes_info if "name" in node_info}

    # Initialize children list for all nodes
    for node_info in nodes_map.values():
        node_info["children"] = []

    # Link children to parents based on 'parent_names'
    for node_info in nodes_map.values():
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

    return [node_info for node_name, node_info in nodes_map.items() if node_name not in all_child_names]


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
    print("\n--- Building Workflow Structure ---")

    if not flat_functionality_dicts:
        print("   Skipping structure building: No functionalities found.")
        return []

    func_list_str = "\n".join(
        [
            f"- Name: {f.get('name', 'N/A')}\n  Description: {f.get('description', 'N/A')}\n  Parameters: {', '.join(p.get('name', '?') for p in f.get('parameters', [])) or 'None'}"
            for f in flat_functionality_dicts
        ],
    )

    snippets = []
    total_snippet_length = 0
    max_total_snippet_length = 7000  # Larger context for structure analysis

    if isinstance(conversation_history, list):
        for i, session_history in enumerate(conversation_history):
            if not isinstance(session_history, list):
                continue

            from chatbot_explorer.conversation.conversation_utils import format_conversation

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

    if chatbot_type == "transactional":
        print("   Using TRANSACTIONAL structuring prompt.")
        structuring_prompt = create_transactional_prompt(func_list_str, conversation_snippets)
    else:  # Default to informational
        print("   Using INFORMATIONAL structuring prompt.")
        structuring_prompt = create_informational_prompt(func_list_str, conversation_snippets)

    try:
        print("   Asking LLM to determine workflow structure...")
        response = llm.invoke(structuring_prompt)
        response_content = response.content

        json_str = extract_json_from_response(response_content)

        # Clean up potential JSON issues
        json_str = re.sub(r"//.*?(\n|$)", "\n", json_str)  # Remove comments
        json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)  # Remove trailing commas

        structured_nodes_info = _parse_and_validate_json_list(json_str)

        root_nodes_dicts = build_node_hierarchy(structured_nodes_info)
        print(f"   Built structure with {len(root_nodes_dicts)} root node(s).")

    except (json.JSONDecodeError, TypeError) as e:  # Catch TypeError from helper
        print(f"   Error: Failed to parse or validate JSON from LLM response: {e}")
        return flat_functionality_dicts  # Return original list on failure
    else:
        return root_nodes_dicts
