from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from .nodes.conversation_params_node import conversation_params_node
from .nodes.goal_generator_node import goal_generator_node
from .nodes.profile_builder_node import profile_builder_node
from .nodes.profile_validator_node import profile_validator_node
from .nodes.structure_builder_node import structure_builder_node
from .state import State


class ChatbotExplorationAgent:
    """Uses LangGraph to explore chatbots and orchestrate analysis."""

    def __init__(self, model_name: str) -> None:
        """Sets up the explorer.

        Args:
            model_name (str): The name of the OpenAI model to use.
        """
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()

    def _build_profile_generation_graph(self) -> StateGraph:
        """Builds a smaller graph just for generating profiles from existing structure.

        Returns:
            StateGraph: The compiled LangGraph application for profile generation.
        """
        graph_builder = StateGraph(State)

        graph_builder.add_node("goal_generator", lambda state: goal_generator_node(state, self.llm))
        graph_builder.add_node(
            "conversation_params",
            lambda state: conversation_params_node(state, self.llm),
        )
        graph_builder.add_node("profile_builder", profile_builder_node)  # Doesn't need llm
        graph_builder.add_node("profile_validator", lambda state: profile_validator_node(state, self.llm))

        # Define the flow
        graph_builder.set_entry_point("goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        # Compile the graph
        return graph_builder.compile(checkpointer=self.memory)

    def _build_structure_graph(self) -> StateGraph:
        """Builds a graph that only runs the structure building part.

        Returns:
            StateGraph: The compiled LangGraph application for structure building.
        """
        graph_builder = StateGraph(State)

        # Add only the structure builder node, passing self.llm
        graph_builder.add_node("structure_builder", lambda state: structure_builder_node(state, self.llm))

        # Start and end with this node
        graph_builder.set_entry_point("structure_builder")
        graph_builder.set_finish_point("structure_builder")

        # Compile the graph
        return graph_builder.compile(checkpointer=self.memory)
