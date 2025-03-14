from typing import Annotated, Dict, List, Any
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from .nodes.goals_node import generate_user_profiles_and_goals
from .nodes.analyzer_node import analyze_conversations


class State(TypedDict):
    """State for the LangGraph flow"""

    messages: Annotated[list, add_messages]
    conversation_history: list
    discovered_functionalities: list
    discovered_limitations: list
    current_session: int
    exploration_finished: bool
    conversation_goals: list
    supported_languages: list


class ChatbotExplorer:
    """Manages exploration of target chatbots using LangGraph."""

    def __init__(self, model_name: str):
        """Initialize explorer with the given model name."""
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph for exploration."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("explorer", self._explorer_node)
        graph_builder.add_node("analyzer", self._analyzer_node)
        graph_builder.add_node("goal_generator", self._goal_generator_node)

        # Add edges
        graph_builder.set_entry_point("explorer")
        graph_builder.add_edge("explorer", "analyzer")
        graph_builder.add_edge("analyzer", "goal_generator")
        graph_builder.set_finish_point("goal_generator")

        return graph_builder.compile(checkpointer=self.memory)

    def _explorer_node(self, state: State):
        """Explorer node for interacting with the target chatbot."""
        if not state["exploration_finished"]:
            return {"messages": [self.llm.invoke(state["messages"])], "explored": True}
        return {"messages": state["messages"]}

    def _analyzer_node(self, state: State):
        """Analyzer node for processing chatbot responses."""
        if state["exploration_finished"]:
            # Use the analyzer module for analysis
            analysis_result = analyze_conversations(
                state["conversation_history"], state["supported_languages"], self.llm
            )

            return {
                "messages": state["messages"] + [analysis_result["analysis_result"]],
                "discovered_functionalities": analysis_result["functionalities"],
                "discovered_limitations": analysis_result["limitations"],
            }
        return {"messages": state["messages"]}

    def _goal_generator_node(self, state: State):
        """Node for generating conversation goals based on discovered functionalities."""
        if state["exploration_finished"] and state["discovered_functionalities"]:
            print("\n--- Generating conversation goals ---")

            profiles_with_goals = generate_user_profiles_and_goals(
                state["discovered_functionalities"],
                state["discovered_limitations"],
                self.llm,
                conversation_history=state["conversation_history"],
                supported_languages=state["supported_languages"],
            )

            return {
                "messages": state["messages"],
                "conversation_goals": profiles_with_goals,
            }
        return {"messages": state["messages"]}

    def run_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Run graph with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.invoke(state, config=config)

    def stream_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream graph execution with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.stream(state, config=config)
