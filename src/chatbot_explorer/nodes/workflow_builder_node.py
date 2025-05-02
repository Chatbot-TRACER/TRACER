"""LangGraph Node that builds the workflow structure from functionalities and conversation history."""

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
    """Count the total number of nodes in a nested list structure.

    Recursively traverses a list of nodes, counting each node and all its children.

    Args:
        nodes_list (list): A list of node dictionaries, where each node may contain
                          a 'children' key with a list of child nodes.

    Returns:
        int: The total count of all nodes in the hierarchy.

    Example:
        >>> nodes = [{"id": 1, "children": [{"id": 2}]}, {"id": 3}]
        >>> count_all_nodes(nodes)
        3
    """
    total = 0
    for node in nodes_list:
        total += 1
        children = node.get("children", [])
        total += count_all_nodes(children)
    return total


def get_workflow_paths(nodes: list[dict], prefix: str = "") -> list[str]:
    """Recursively generates a list of workflow paths from a hierarchical node structure.

    This function traverses a tree of nodes and creates formatted string representations
    of each path in the workflow, showing the parent-child relationships. Endpoints
    (nodes without children) are marked as such.

    Args:
        nodes (list): List of node dictionaries, where each node may contain:
                     - name: The node's name (default: "unnamed")
                     - description: The node's description (not used in output)
                     - children: List of child nodes (optional)
        prefix (str, optional): String prefix for indentation in recursive calls.
                               Defaults to "".

    Returns:
        list[str]: A list of formatted path strings, where:
                  - Parent nodes show their name followed by the names of up to 3 children
                  - If a node has more than 3 children, additional children are summarized
                  - Child paths are indented with "  " per level of depth
                  - Nodes without children are marked as "(endpoint)"

    Example:
        For a node structure with parent "A" and children "B", "C", and "D",
        the function would return:
        ["A → B, C, D", "  B (endpoint)", "  C (endpoint)", "  D (endpoint)"]
    """
    paths = []
    for node in nodes:
        node_name = node.get("name", "unnamed")
        node.get("description", "").split(".")[0]  # First sentence of description
        children = node.get("children", [])
        if children:
            child_names = ", ".join([child.get("name", "unnamed") for child in children[:MAX_CHILDREN_DISPLAYED]])
            if len(children) > MAX_CHILDREN_DISPLAYED:
                child_names += f", +{len(children) - MAX_CHILDREN_DISPLAYED} more"
                child_names += f", +{len(children) - 3} more"
            path_info = f"{prefix}{node_name} → {child_names}"
            paths.append(path_info)
            # Add child paths with indentation
            paths.extend(get_workflow_paths(children, prefix=f"  {prefix}"))
        else:
            paths.append(f"{prefix}{node_name} (endpoint)")
    return paths


def workflow_builder_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that analyzes functionalities and history to build the workflow structure."""
    logger.debug("Analyzing workflow structure from discovered functionalities")

    # Functionalities are expected as dicts from run_full_exploration results
    flat_functionality_dicts = state.get("discovered_functionalities", [])
    conversation_history = state.get("conversation_history", [])

    if not flat_functionality_dicts:
        logger.warning("Skipping structure building: No initial functionalities found")
        # Return partial state update
        return {
            "discovered_functionalities": [],
            "chatbot_type": "unknown",  # Ensure type is set
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

        # Enhanced logging with more information about the structure
        node_count = len(structured_nodes)

        # Count total nodes including children at all levels

        total_nodes = count_all_nodes(structured_nodes)

        # Get information about workflow structure

        workflow_paths = get_workflow_paths(structured_nodes)

        logger.info("Workflow structure created with %d root nodes and %d total nodes", node_count, total_nodes)

        if workflow_paths:
            logger.info("\nWorkflow structure:")
            for path in workflow_paths[:MAX_WORKFLOW_PATHS]:  # Limit to first 10 paths to avoid overwhelming logs
                logger.info(" • %s", path)
            if len(workflow_paths) > MAX_WORKFLOW_PATHS:
                logger.info(" • ... and %d more paths", len(workflow_paths) - MAX_WORKFLOW_PATHS)

        logger.debug("Root node names: %s", ", ".join([node.get("name", "unnamed") for node in structured_nodes]))

    except (ValueError, KeyError, TypeError):
        # Handle errors during structure building
        logger.exception("Error during structure building")
        # Keep the original flat list but update bot_type
        return {
            "discovered_functionalities": flat_functionality_dicts,
            "chatbot_type": bot_type,
        }
    else:
        # Update state with the final structured list of dictionaries and bot type
        return {
            "discovered_functionalities": structured_nodes,
            "chatbot_type": bot_type,
        }
