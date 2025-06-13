"""LangGraph Node that builds the workflow structure from functionalities and conversation history."""

import json
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.analysis.chatbot_classification import classify_chatbot_type
from tracer.analysis.workflow_builder import build_workflow_structure
from tracer.schemas.graph_state_model import State
from tracer.utils.logging_utils import get_logger

logger = get_logger()
MAX_WORKFLOW_PATHS = 10
MAX_CHILDREN_DISPLAYED = 3


def count_all_nodes(nodes_list: list[dict]) -> int:
    """Count the total number of nodes in a nested list of dictionaries.

    Recursively traverses a list of node dictionaries, counting each node
    and all its children. Avoids double-counting nodes visited via different paths
    in a DAG by keeping track of visited node names.

    Args:
        nodes_list (list): A list of node dictionaries, where each node may contain
                          a 'children' key with a list of child node dictionaries.

    Returns:
        int: The total count of unique nodes in the hierarchy.
    """
    # Use a set to track visited nodes to handle DAGs correctly
    visited_node_names = set()
    nodes_to_process = list(nodes_list)  # Start with root nodes

    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        node_name = current_node.get("name")

        # Process node only if it has a name and hasn't been visited
        if node_name and node_name not in visited_node_names:
            visited_node_names.add(node_name)
            children = current_node.get("children", [])
            if isinstance(children, list):
                # Add children to the processing queue
                nodes_to_process.extend(children)

    return len(visited_node_names)


def get_workflow_paths(nodes: list[dict], prefix: str = "", visited_nodes: set | None = None) -> list[str]:
    """Recursively generates a list of workflow paths from a hierarchical node structure.

    Handles potential cycles/DAGs by tracking visited nodes in the current path to prevent infinite loops.

    Args:
        nodes (list): List of node dictionaries.
        prefix (str, optional): String prefix for indentation. Defaults to "".
        visited_nodes (set, optional): A set to track visited node names in the current path
                                      to avoid infinite loops. Should be initialized as None
                                      in the top-level call.

    Returns:
        list[str]: A list of formatted path strings.
    """
    if visited_nodes is None:
        visited_nodes = set()

    paths = []

    for node in nodes:
        node_name = node.get("name", "")
        current_path = _build_current_path(node, prefix)

        # Check for circular reference using node name (not path)
        if node_name and node_name in visited_nodes:
            # This is a back-reference - add a marker and continue without recursing
            paths.append(f"{current_path} (*)")
            continue

        # Add this node to visited set for this path
        if node_name:
            visited_nodes.add(node_name)

        if _has_children(node):
            # Recursively get child paths with the current visited_nodes set
            child_paths = get_workflow_paths(node["children"], current_path, visited_nodes.copy())
            paths.extend(child_paths)
        else:
            paths.append(current_path)

        # Remove this node from visited set when backtracking
        if node_name:
            visited_nodes.discard(node_name)

    return paths


def _build_current_path(node: dict, prefix: str) -> str:
    """Build the current path from node name and prefix."""
    name = node.get("name", "Unknown")
    return f"{prefix}/{name}" if prefix else name


def _has_children(node: dict) -> bool:
    """Check if node has children."""
    return "children" in node and node["children"]


def workflow_builder_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that analyzes functionalities and history to build the workflow structure."""
    logger.debug("Analyzing workflow structure from discovered functionalities")

    # Functionalities are expected as dicts from run_full_exploration results
    flat_functionality_dicts = state.get("discovered_functionalities", [])
    conversation_history = state.get("conversation_history", [])

    if not flat_functionality_dicts:
        logger.warning("Skipping structure building: No initial functionalities found")
        return {
            "discovered_functionalities": [],
            "chatbot_type": "unknown",
        }

    # Classify the bot type first
    logger.info("=== Classifying Chatbot ===")
    bot_type = classify_chatbot_type(flat_functionality_dicts, conversation_history, llm)
    logger.info("Chatbot type classified as: %s", bot_type)

    try:
        logger.debug(
            "Building workflow structure based on %d discovered functionalities", len(flat_functionality_dicts)
        )

        structured_nodes = build_workflow_structure(flat_functionality_dicts, conversation_history, bot_type, llm)

        root_node_count = len(structured_nodes)
        total_unique_nodes = count_all_nodes(structured_nodes)

        logger.info(
            "Workflow structure created with %d root nodes and %d unique total nodes",
            root_node_count,
            total_unique_nodes,
        )

        # Log the paths using the modified path generator
        # Pass None for visited_nodes initially
        workflow_paths = get_workflow_paths(structured_nodes, visited_nodes=None)
        if workflow_paths:
            logger.info(
                "\nWorkflow structure (paths shown once per parent; (*) indicates node visited via another path):"
            )
            for path in workflow_paths[:MAX_WORKFLOW_PATHS]:
                logger.info(" • %s", path)
            if len(workflow_paths) > MAX_WORKFLOW_PATHS:
                logger.info(" • ... and %d more paths", len(workflow_paths) - MAX_WORKFLOW_PATHS)
        else:
            logger.info("No workflow paths generated from the structure.")

        logger.debug("Root node names: %s", ", ".join([node.get("name", "unnamed") for node in structured_nodes]))
        logger.info(
            "Note: The final JSON output accurately represents joins via the 'parent_names' field, even if paths appear duplicated in logs."
        )

    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        logger.exception("Error during structure building or processing")
        return {
            "discovered_functionalities": flat_functionality_dicts,  # Keep original flat list
            "chatbot_type": bot_type,
        }
    else:
        # Update state with the final structured list of dictionaries and bot type
        return {
            "discovered_functionalities": structured_nodes,  # This is the dict hierarchy
            "chatbot_type": bot_type,
        }
