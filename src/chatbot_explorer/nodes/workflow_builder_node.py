"""LangGraph Node that builds the workflow structure from functionalities and conversation history."""

import json
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.analysis.chatbot_classification import classify_chatbot_type
from chatbot_explorer.analysis.workflow_builder import build_workflow_structure
from chatbot_explorer.schemas.graph_state_model import State
from chatbot_explorer.utils.logging_utils import get_logger

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


def get_workflow_paths(nodes: list[dict], prefix: str = "", visited_paths: set = None) -> list[str]:
    """Recursively generates a list of workflow paths from a hierarchical node structure.

    Handles potential cycles/DAGs by tracking visited parent-child relationships for display.

    Args:
        nodes (list): List of node dictionaries.
        prefix (str, optional): String prefix for indentation. Defaults to "".
        visited_paths (set, optional): A set to track visited (parent_name, child_name)
                                       tuples to avoid infinite loops in logs for DAGs.
                                       Should be initialized as None in the top-level call.

    Returns:
        list[str]: A list of formatted path strings.
    """
    if visited_paths is None:
        visited_paths = set()

    paths = []
    for node in nodes:
        node_name = node.get("name", "unnamed")
        children = node.get("children", [])

        # Format current node line
        if children:
            child_names_list = []
            for child in children[:MAX_CHILDREN_DISPLAYED]:
                child_name = child.get("name", "unnamed")
                # Check if this specific parent->child path has been logged before
                path_tuple = (node_name, child_name)
                if path_tuple in visited_paths:
                    child_names_list.append(f"{child_name} (*)")  # Mark as already shown path
                else:
                    child_names_list.append(child_name)

            child_names = ", ".join(child_names_list)

            if len(children) > MAX_CHILDREN_DISPLAYED:
                child_names += f", +{len(children) - MAX_CHILDREN_DISPLAYED} more"
            path_info = f"{prefix}{node_name} → {child_names}"
        else:
            path_info = f"{prefix}{node_name} (endpoint)"

        paths.append(path_info)

        # Recursively add child paths only if not visited before in this traversal
        if children:
            child_paths_to_add = []
            current_children_nodes = []
            for child in children:
                child_name = child.get("name", "unnamed")
                path_tuple = (node_name, child_name)
                if path_tuple not in visited_paths:
                    visited_paths.add(path_tuple)  # Mark this specific path as visited for display
                    current_children_nodes.append(child)  # Only recurse through unvisited paths

            if current_children_nodes:
                # Pass the *same* visited_paths set down
                child_paths_to_add.extend(
                    get_workflow_paths(current_children_nodes, prefix=f"  {prefix}", visited_paths=visited_paths)
                )

            paths.extend(child_paths_to_add)

    return paths


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
        # Pass None for visited_paths initially
        workflow_paths = get_workflow_paths(structured_nodes, visited_paths=None)
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
