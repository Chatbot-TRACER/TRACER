"""ChatbotExplorer Agent, contains methods to run the exploration and analysis."""

import uuid
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chatbot_explorer.conversation.fallback_detection import extract_fallback_message
from chatbot_explorer.conversation.language_detection import extract_supported_languages
from chatbot_explorer.conversation.session import (
    ExplorationGraphState,
    ExplorationSessionConfig,
    run_exploration_session,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode
from connectors.chatbot_connectors import Chatbot

from .graphs.profile_graph import build_profile_generation_graph
from .graphs.structure_graph import build_structure_graph
from .schemas.graph_state_model import State


class ExplorationParams(TypedDict):
    """Parameters for running exploration sessions."""

    max_sessions: int
    max_turns: int
    supported_languages: list[str]
    fallback_message: str | None


class SessionParams(TypedDict):
    """Parameters specific to a single exploration session."""

    session_num: int
    max_sessions: int
    max_turns: int


class ChatbotExplorationAgent:
    """Uses LangGraph to explore chatbots and orchestrate analysis."""

    def __init__(self, model_name: str) -> None:
        """Sets up the explorer.

        Args:
            model_name (str): The name of the OpenAI model to use.
        """
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()

        self._structure_graph = build_structure_graph(self.llm, self.memory)
        self._profile_graph = build_profile_generation_graph(self.llm, self.memory)

    def run_exploration(self, chatbot_connector: Chatbot, max_sessions: int, max_turns: int) -> dict[str, Any]:
        """Runs the initial probing and the main exploration loop.

        Args:
            chatbot_connector: An instance of a chatbot connector class.
            max_sessions: Maximum number of exploration sessions to run.
            max_turns: Maximum turns per exploration session.

        Returns:
            Dictionary containing exploration results (conversation histories,
            functionality nodes, supported languages, and fallback message).
        """
        # Initialize results storage
        conversation_sessions = []
        current_graph_state = self._initialize_graph_state()

        # Perform initial probing steps
        supported_languages = self._detect_languages(chatbot_connector)
        fallback_message = self._detect_fallback(chatbot_connector)

        # Create exploration parameters
        exploration_params: ExplorationParams = {
            "max_sessions": max_sessions,
            "max_turns": max_turns,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

        # Run the main exploration sessions
        conversation_sessions, current_graph_state = self._run_exploration_sessions(
            chatbot_connector, exploration_params, current_graph_state
        )

        # Convert final root nodes to dictionaries for the result
        functionality_dicts = [node.to_dict() for node in current_graph_state["root_nodes"]]

        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

    def _initialize_graph_state(self) -> ExplorationGraphState:
        """Initialize the graph state for exploration."""
        return {
            "root_nodes": [],
            "pending_nodes": [],
            "explored_nodes": set(),
        }

    def _detect_languages(self, chatbot_connector: Chatbot) -> list[str]:
        """Detect languages supported by the chatbot."""
        print("\n--- Probing Chatbot Language ---")
        initial_probe_query = "Hello"
        is_ok, probe_response = chatbot_connector.execute_with_input(initial_probe_query)

        supported_languages = ["English"]  # Default

        if is_ok and probe_response:
            print(f"   Initial response received: '{probe_response[:60]}...'")
            try:
                detected_langs = extract_supported_languages(probe_response, self.llm)
                if detected_langs:
                    supported_languages = detected_langs
                    print(f"   Detected initial language(s): {supported_languages}")
                else:
                    print("   Could not detect language from initial probe, defaulting to English.")
            except (ValueError, TypeError) as lang_e:
                print(f"   Error during initial language detection: {lang_e}. Defaulting to English.")
        else:
            print("   Could not get initial response from chatbot for language probe. Defaulting to English.")

        return supported_languages

    def _detect_fallback(self, chatbot_connector: Chatbot) -> str | None:
        """Detect the chatbot's fallback message."""
        print("\n--- Attempting to detect chatbot fallback message ---")
        fallback_message = None

        try:
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
            if fallback_message:
                print(f'   Detected fallback message: "{fallback_message[:50]}..."')
            else:
                print("   Could not detect a fallback message.")
        except (ValueError, KeyError, AttributeError) as fb_e:
            print(f"   Error during fallback detection: {fb_e}. Proceeding without fallback.")

        return fallback_message

    def _run_exploration_sessions(
        self,
        chatbot_connector: Chatbot,
        params: ExplorationParams,
        graph_state: ExplorationGraphState,
    ) -> tuple[list[list[dict[str, str]]], ExplorationGraphState]:
        """Run multiple exploration sessions with the chatbot.

        Args:
            chatbot_connector: Chatbot to interact with
            params: Exploration parameters including max sessions, turns, languages and fallback
            graph_state: Current state of the exploration graph

        Returns:
            Tuple of conversation sessions and updated graph state
        """
        conversation_sessions = []
        current_graph_state = graph_state

        session_num = 0
        while session_num < params["max_sessions"]:
            print(f"\n=== Starting Session {session_num + 1}/{params['max_sessions']} ===")

            # Determine which node to explore next
            explore_node, session_type = self._select_next_node(current_graph_state, session_num)

            # Skip already explored nodes
            if explore_node and explore_node.name in current_graph_state["explored_nodes"]:
                print(f"--- Skipping already explored node: '{explore_node.name}' ---")
                session_num += 1
                continue

            print(f"   Session Type: {session_type}")

            # Configure and run a single session
            session_params: SessionParams = {
                "session_num": session_num,
                "max_sessions": params["max_sessions"],
                "max_turns": params["max_turns"],
            }

            session_config = self._create_session_config(
                session_params,
                chatbot_connector,
                params,
                explore_node,
                current_graph_state,
            )

            conversation_history, updated_graph_state = run_exploration_session(
                config=session_config,
            )

            conversation_sessions.append(conversation_history)
            current_graph_state = updated_graph_state
            session_num += 1

        # Display summary information
        self._print_exploration_summary(session_num, current_graph_state)

        return conversation_sessions, current_graph_state

    def _select_next_node(self, graph_state: ExplorationGraphState, session_num: int) -> tuple[Any, str]:
        """Select the next node to explore."""
        explore_node = None
        session_type = "General Exploration"

        if graph_state["pending_nodes"]:
            explore_node = graph_state["pending_nodes"].pop(0)
            session_type = f"Exploring functionality '{explore_node.name}'"
        elif session_num > 0:
            print("   Pending nodes queue is empty. Performing general exploration.")

        return explore_node, session_type

    def _create_session_config(
        self,
        session_params: SessionParams,
        chatbot_connector: Chatbot,
        exploration_params: ExplorationParams,
        current_node: FunctionalityNode,
        graph_state: ExplorationGraphState,
    ) -> ExplorationSessionConfig:
        """Create a configuration for a single exploration session.

        Args:
            session_params: Session-specific parameters (number, max sessions, turns)
            chatbot_connector: The chatbot connector instance
            exploration_params: General exploration parameters
            current_node: The current node being explored (if any)
            graph_state: Current state of the exploration graph

        Returns:
            Configuration dictionary for the session
        """
        return {
            "session_num": session_params["session_num"],
            "max_sessions": session_params["max_sessions"],
            "max_turns": session_params["max_turns"],
            "llm": self.llm,
            "the_chatbot": chatbot_connector,
            "fallback_message": exploration_params["fallback_message"],
            "current_node": current_node,
            "graph_state": graph_state,
            "supported_languages": exploration_params["supported_languages"],
        }

    def _print_exploration_summary(self, session_count: int, graph_state: ExplorationGraphState) -> None:
        """Print summary information after exploration."""
        print(f"\n=== Completed {session_count} exploration sessions ===")

        if graph_state["pending_nodes"]:
            print(f"   NOTE: {len(graph_state['pending_nodes'])} nodes still remain in the pending queue.")
        else:
            print("   All discovered nodes were explored.")

        print(f"Discovered {len(graph_state['root_nodes'])} root functionalities after exploration.")

    def run_analysis(self, exploration_results: dict[str, Any]) -> dict[str, list[Any]]:
        """Runs the LangGraph analysis pipeline using pre-compiled graphs.

        Args:
            exploration_results: A dictionary containing results from the exploration phase.

        Returns:
            A dictionary containing 'discovered_functionalities' and 'built_profiles'.
        """
        print("\n--- Preparing for analysis phase ---")

        # 1. Prepare initial state for the structure graph
        structure_initial_state = State(
            messages=[{"role": "system", "content": "Infer structure from conversation history."}],
            conversation_history=exploration_results.get("conversation_sessions", []),
            discovered_functionalities=exploration_results.get("root_nodes_dict", {}),  # Use the dict format
            built_profiles=[],
            discovered_limitations=[],
            current_session=len(exploration_results.get("conversation_sessions", [])),
            exploration_finished=True,
            conversation_goals=[],
            supported_languages=exploration_results.get("supported_languages", []),
            fallback_message=exploration_results.get("fallback_message", ""),
            workflow_structure=None,
            # Initialize any other State fields with defaults if necessary
        )

        # -- Run Structure Inference --
        print("\n--- Running workflow structure inference ---")
        structure_thread_id = f"structure_analysis_{uuid.uuid4()}"
        structure_result = self._structure_graph.invoke(
            structure_initial_state,
            config={"configurable": {"thread_id": structure_thread_id}},
        )
        workflow_structure = structure_result.get("discovered_functionalities", {})
        print("--- Structure inference complete ---")

        # 2. Prepare initial state for the profile graph
        profile_initial_state = structure_result.copy()
        profile_initial_state["workflow_structure"] = workflow_structure
        profile_initial_state["messages"] = [
            {"role": "system", "content": "Generate user profiles based on the workflow structure."},
        ]
        profile_initial_state["conversation_goals"] = []
        profile_initial_state["built_profiles"] = []

        # -- Run Profile Generation --
        print("\n--- Generating user profiles ---")
        profile_thread_id = f"profile_analysis_{uuid.uuid4()}"
        profile_result = self._profile_graph.invoke(
            profile_initial_state,
            config={"configurable": {"thread_id": profile_thread_id}},
        )

        generated_profiles = profile_result.get("built_profiles", [])
        print(f"--- Analysis complete, {len(generated_profiles)} profiles generated ---")

        return {
            "discovered_functionalities": workflow_structure,
            "built_profiles": generated_profiles,
        }
