"""LangGraph Node that builds the workflow structure from functionalities and conversation history."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.analysis.chatbot_classification import classify_chatbot_type
from chatbot_explorer.analysis.workflow_builder import build_workflow_structure
from chatbot_explorer.schemas.graph_state_model import State
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()


def count_all_nodes(nodes_list) -> int:
    total = 0
    for node in nodes_list:
        total += 1
        children = node.get("children", [])
        total += count_all_nodes(children)
    return total


def get_workflow_paths(nodes, prefix="") -> list[str]:
    paths = []
    for node in nodes:
        node_name = node.get("name", "unnamed")
        node_desc = node.get("description", "").split(".")[0]  # First sentence of description
        children = node.get("children", [])

        if children:
            child_names = ", ".join([child.get("name", "unnamed") for child in children[:3]])
            if len(children) > 3:
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
    logger.info("Analyzing workflow structure from discovered functionalities")

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
    logger.info("Classifying chatbot interaction type")
    bot_type = classify_chatbot_type(flat_functionality_dicts, conversation_history, llm)
    logger.info("Chatbot type classified as: %s", bot_type)

    try:
        logger.info("Building workflow structure based on %d discovered functionalities", len(flat_functionality_dicts))
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
            for path in workflow_paths[:10]:  # Limit to first 10 paths to avoid overwhelming logs
                logger.info(" • %s", path)
            if len(workflow_paths) > 10:
                logger.info(" • ... and %d more paths", len(workflow_paths) - 10)

        logger.debug("Root node names: %s", ", ".join([node.get("name", "unnamed") for node in structured_nodes]))

    except (ValueError, KeyError, TypeError) as e:
        # Handle errors during structure building
        logger.error("Error during structure building: %s", str(e))
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
