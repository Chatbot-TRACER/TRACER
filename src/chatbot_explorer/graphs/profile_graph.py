from langchain_core.language_models import BaseLanguageModel  # Use base class
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END

from ..nodes.goal_generator_node import goal_generator_node
from ..nodes.conversation_params_node import conversation_params_node
from ..nodes.profile_builder_node import profile_builder_node
from ..nodes.profile_validator_node import profile_validator_node

from ..schemas.state import State


def build_profile_generation_graph(llm: BaseLanguageModel, checkpointer: BaseCheckpointSaver):
    """
    Builds and compiles the LangGraph for generating user profiles.

    Args:
        llm: The language model instance to be used by nodes.
        checkpointer: The checkpointer instance for saving graph state.

    Returns:
        A compiled LangGraph application (Runnable).
    """
    graph_builder = StateGraph(State)

    # Add nodes, passing LLM where needed via lambda
    graph_builder.add_node("goal_generator", lambda state: goal_generator_node(state, llm))
    graph_builder.add_node("conversation_params", lambda state: conversation_params_node(state, llm))

    graph_builder.add_node("profile_builder", profile_builder_node)
    graph_builder.add_node("profile_validator", lambda state: profile_validator_node(state, llm))

    graph_builder.set_entry_point("goal_generator")
    graph_builder.add_edge("goal_generator", "conversation_params")
    graph_builder.add_edge("conversation_params", "profile_builder")
    graph_builder.add_edge("profile_builder", "profile_validator")

    graph_builder.add_edge("profile_validator", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph
