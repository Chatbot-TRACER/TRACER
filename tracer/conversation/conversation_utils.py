"""Utility functions for formatting conversations and handling functionality nodes."""

from tracer.schemas.functionality_node_model import FunctionalityNode


def format_conversation(messages: list[dict[str, str]]) -> str:
    """Make the conversation history easy to read.

    Args:
        messages (List[Dict[str, str]]): The list of message dictionaries.

    Returns:
        str: A formatted string representing the conversation.
    """
    formatted = []
    for msg in messages:
        if msg["role"] in ["assistant", "user"]:
            # 'assistant' is our explorer AI, 'user' is the chatbot being tested
            sender = "Human" if msg["role"] == "assistant" else "Chatbot"
            formatted.append(f"{sender}: {msg['content']}")
    return "\n".join(formatted)


def _get_all_nodes(root_node: FunctionalityNode) -> list:
    """Helper to get a flat list of nodes in a tree.

    Args:
        root_node: The starting node of the tree/subtree.

    Returns:
        list: A flat list containing the root_node and all its descendants.
    """
    result = [root_node]
    for child in root_node.children:
        result.extend(_get_all_nodes(child))  # Recursive call
    return result
