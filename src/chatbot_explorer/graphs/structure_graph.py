# src/chatbot_explorer/graphs/structure_graph.py

from langchain_core.language_models import BaseLanguageModel  # Use base class for flexibility
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END

# Import the specific node function and the State definition
# Adjust the relative path based on your exact structure if needed
from ..nodes.structure_builder_node import structure_builder_node

from ..state import State

def build_structure_graph(llm: BaseLanguageModel, checkpointer: BaseCheckpointSaver):
    """
    Builds and compiles the LangGraph for inferring workflow structure.

    Args:
        llm: The language model instance to be used by nodes.
        checkpointer: The checkpointer instance for saving graph state.

    Returns:
        A compiled LangGraph application (Runnable).
    """
    graph_builder = StateGraph(State)

    # Add the structure builder node
    # Pass llm if needed by the node, otherwise just pass the node function
    graph_builder.add_node(
        "structure_builder",
        lambda state: structure_builder_node(state, llm),  # Assuming node needs llm
        # or just: structure_builder_node # If node doesn't need llm
    )

    # Define the flow (simple in this case)
    graph_builder.set_entry_point("structure_builder")
    # Use END for clarity as the terminal state
    graph_builder.add_edge("structure_builder", END)

    # Compile the graph
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph
