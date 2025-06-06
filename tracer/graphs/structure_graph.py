"""Module to generate the graph for workflow structure inference."""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from tracer.nodes.workflow_builder_node import workflow_builder_node
from tracer.schemas.graph_state_model import State


def build_structure_graph(llm: BaseLanguageModel, checkpointer: BaseCheckpointSaver) -> Runnable:
    """Builds and compiles the LangGraph for inferring workflow structure.

    Args:
        llm: The language model instance to be used by nodes.
        checkpointer: The checkpointer instance for saving graph state.

    Returns:
        A compiled LangGraph application (Runnable).
    """
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "workflow_builder",
        lambda state: workflow_builder_node(state, llm),
    )

    graph_builder.set_entry_point("workflow_builder")
    graph_builder.add_edge("workflow_builder", END)

    return graph_builder.compile(checkpointer=checkpointer)
