"""Module to generate the graph for user profile generation."""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from tracer.nodes.conversation_params_node import conversation_params_node
from tracer.nodes.profile_builder_node import profile_builder_node
from tracer.nodes.profile_generator_node import profile_generator_node
from tracer.nodes.profile_validator_node import profile_validator_node
from tracer.schemas.graph_state_model import State


def build_profile_generation_graph(llm: BaseLanguageModel, checkpointer: BaseCheckpointSaver) -> Runnable:
    """Builds and compiles the LangGraph for generating user profiles.

    Args:
        llm: The language model instance to be used by nodes.
        checkpointer: The checkpointer instance for saving graph state.

    Returns:
        A compiled LangGraph application (Runnable).
    """
    graph_builder = StateGraph(State)

    # Add nodes, passing LLM where needed via lambda
    graph_builder.add_node("goal_generator", lambda state: profile_generator_node(state, llm))
    graph_builder.add_node("conversation_params", lambda state: conversation_params_node(state, llm))

    graph_builder.add_node("profile_builder", profile_builder_node)
    graph_builder.add_node("profile_validator", lambda state: profile_validator_node(state, llm))

    graph_builder.set_entry_point("goal_generator")
    graph_builder.add_edge("goal_generator", "conversation_params")
    graph_builder.add_edge("conversation_params", "profile_builder")
    graph_builder.add_edge("profile_builder", "profile_validator")

    graph_builder.add_edge("profile_validator", END)

    return graph_builder.compile(checkpointer=checkpointer)
