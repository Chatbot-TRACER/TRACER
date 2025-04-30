from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from chatbot_explorer.nodes.structure_builder_node import structure_builder_node
from chatbot_explorer.schemas.state import State


def build_structure_graph(llm: BaseLanguageModel, checkpointer: BaseCheckpointSaver):
    """Builds and compiles the LangGraph for inferring workflow structure.

    Args:
        llm: The language model instance to be used by nodes.
        checkpointer: The checkpointer instance for saving graph state.

    Returns:
        A compiled LangGraph application (Runnable).
    """
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "structure_builder",
        lambda state: structure_builder_node(state, llm),
    )

    graph_builder.set_entry_point("structure_builder")
    graph_builder.add_edge("structure_builder", END)

    return graph_builder.compile(checkpointer=checkpointer)
