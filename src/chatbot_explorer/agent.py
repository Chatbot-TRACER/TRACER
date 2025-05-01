"""Orchestrates chatbot exploration and analysis using LangGraph."""

import uuid
from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chatbot_explorer.conversation.fallback_detection import extract_fallback_message
from chatbot_explorer.conversation.language_detection import extract_supported_languages
from chatbot_explorer.conversation.session import (
    ExplorationGraphState,
    ExplorationSessionConfig,
    run_exploration_session,
)
from connectors.chatbot_connectors import Chatbot

from .graphs.profile_graph import build_profile_generation_graph
from .graphs.structure_graph import build_structure_graph
from .schemas.graph_state_model import State

if TYPE_CHECKING:
    from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode


class ChatbotExplorationAgent:
    """Uses LangGraph to explore chatbots and orchestrate analysis.

    This agent first runs exploration sessions to interact with a chatbot,
    detecting its language, fallback messages, and mapping its functionalities.
    Then, it uses separate LangGraph graphs to infer the chatbot's structure
    and generate user profiles based on the exploration results.

    Attributes:
        llm: The language model instance used for generation and analysis.
        memory: A memory saver instance for graph state persistence.
    """

    def __init__(self, model_name: str) -> None:
        """Initializes the agent with a specified language model.

        Args:
            model_name: The name of the OpenAI model to use (e.g., "gpt-4o").
        """
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()

        self._structure_graph = build_structure_graph(self.llm, self.memory)
        self._profile_graph = build_profile_generation_graph(self.llm, self.memory)

    def _probe_language(self, chatbot_connector: Chatbot) -> list[str]:
        """Attempts to detect the chatbot's supported languages."""
        print("\n--- Probing Chatbot Language ---")
        initial_probe_query = "Hello"
        supported_languages = ["English"]  # Default
        try:
            is_ok, probe_response = chatbot_connector.execute_with_input(initial_probe_query)
            if is_ok and probe_response:
                print(f"   Initial response received: '{probe_response[:60]}...'")
                detected_langs = extract_supported_languages(probe_response, self.llm)
                if detected_langs:
                    supported_languages = detected_langs
                    print(f"   Detected initial language(s): {supported_languages}")
                else:
                    print("   Could not detect language from initial probe, defaulting to English.")
            else:
                print("   Could not get initial response for language probe. Defaulting to English.")
        # Catch specific, anticipated errors if possible, otherwise broaden
        except (ValueError, KeyError, AttributeError, ConnectionError) as lang_e:
            print(f"   Error during initial language detection: {lang_e}. Defaulting to English.")
        return supported_languages

    def _probe_fallback(self, chatbot_connector: Chatbot) -> str | None:
        """Attempts to detect the chatbot's fallback message."""
        print("\n--- Attempting to detect chatbot fallback message ---")
        fallback_message = None
        try:
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
            if fallback_message:
                print(f'   Detected fallback message: "{fallback_message[:50]}..."')
            else:
                print("   Could not detect a fallback message.")
        # Catch specific, anticipated errors if possible, otherwise broaden
        except (ValueError, KeyError, AttributeError, ConnectionError) as fb_e:
            print(f"   Error during fallback detection: {fb_e}. Proceeding without fallback.")
        return fallback_message

    def _run_single_exploration_session(
        self,
        session_config: ExplorationSessionConfig,
    ) -> tuple[list[dict[str, str]], ExplorationGraphState]:
        """Runs one exploration session and updates the graph state."""
        explore_node = session_config["current_node"]
        session_type_log = f"Exploring functionality '{explore_node.name}'" if explore_node else "General Exploration"
        print(f"   Session Type: {session_type_log}")

        conversation_history, updated_graph_state = run_exploration_session(
            config=session_config,
        )
        return conversation_history, updated_graph_state

    def run_exploration(self, chatbot_connector: Chatbot, max_sessions: int, max_turns: int) -> dict[str, Any]:
        """Runs the initial probing and the main exploration loop.

        Args:
            chatbot_connector: An instance of a chatbot connector class
                               (needs an `execute_with_input` method).
            max_sessions: Maximum number of exploration sessions to run.
            max_turns: Maximum turns per exploration session.

        Returns:
            A dictionary containing exploration results:
                - conversation_sessions: List of conversation histories.
                - root_nodes_dict: List of root FunctionalityNode objects as dicts.
                - supported_languages: List of detected languages.
                - fallback_message: Detected fallback message string.
        """
        conversation_sessions: list[list[dict[str, str]]] = []
        supported_languages = self._probe_language(chatbot_connector)
        fallback_message = self._probe_fallback(chatbot_connector)

        current_graph_state: ExplorationGraphState = {
            "root_nodes": [],
            "pending_nodes": [],
            "explored_nodes": set(),
        }

        session_num = 0
        while session_num < max_sessions:
            current_session_index = session_num
            print(f"\n=== Starting Session {current_session_index + 1}/{max_sessions} ===")

            explore_node: FunctionalityNode | None = None
            if current_graph_state["pending_nodes"]:
                next_node = current_graph_state["pending_nodes"].pop(0)
                if next_node.name in current_graph_state["explored_nodes"]:
                    print(f"--- Skipping already explored node: '{next_node.name}' ---")
                    continue
                explore_node = next_node
            elif session_num > 0:
                print("   Pending nodes queue is empty. Performing general exploration.")

            # Pass the current state into the config for the session runner
            session_config: ExplorationSessionConfig = {
                "session_num": current_session_index,
                "max_sessions": max_sessions,
                "max_turns": max_turns,
                "llm": self.llm,
                "the_chatbot": chatbot_connector,
                "fallback_message": fallback_message,
                "current_node": explore_node,
                "graph_state": current_graph_state,
                "supported_languages": supported_languages,
            }

            conversation_history, updated_graph_state = self._run_single_exploration_session(session_config)

            conversation_sessions.append(conversation_history)
            current_graph_state = updated_graph_state
            session_num += 1

        # --- Post-Exploration Summary ---
        print(f"\n=== Completed {session_num} exploration sessions ===")
        if current_graph_state["pending_nodes"]:
            print(f"   NOTE: {len(current_graph_state['pending_nodes'])} nodes still remain in the pending queue.")
        else:
            print("   All discovered nodes were explored.")
        print(f"Discovered {len(current_graph_state['root_nodes'])} root functionalities after exploration.")

        functionality_dicts = [node.to_dict() for node in current_graph_state["root_nodes"]]

        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

    def run_analysis(self, exploration_results: dict[str, Any]) -> dict[str, list[Any]]:
        """Runs the LangGraph analysis pipeline using pre-compiled graphs.

        Args:
            exploration_results: A dictionary containing results from the exploration phase.

        Returns:
            A dictionary containing 'discovered_functionalities' (structured)
            and 'built_profiles'.
        """
        print("\n--- Preparing for analysis phase ---")

        # Prepare initial state for the structure graph
        structure_initial_state = State(
            messages=[{"role": "system", "content": "Infer structure from conversation history."}],
            conversation_history=exploration_results.get("conversation_sessions", []),
            discovered_functionalities=exploration_results.get("root_nodes_dict", []),  # Use dict list
            built_profiles=[],
            discovered_limitations=[],
            current_session=len(exploration_results.get("conversation_sessions", [])),
            exploration_finished=True,
            conversation_goals=[],
            supported_languages=exploration_results.get("supported_languages", []),
            fallback_message=exploration_results.get("fallback_message", ""),
            workflow_structure=None,
            chatbot_type="unknown",  # Initialize chatbot_type
        )

        # Run Structure Inference
        print("\n--- Running workflow structure inference ---")
        structure_thread_id = f"structure_analysis_{uuid.uuid4()}"
        structure_result = self._structure_graph.invoke(
            structure_initial_state,
            config={"configurable": {"thread_id": structure_thread_id}},
        )
        workflow_structure = structure_result.get("discovered_functionalities", [])
        print("--- Structure inference complete ---")

        # Prepare initial state for the profile graph
        # Use the result from the structure graph as the starting point
        profile_initial_state = structure_result.copy()
        # Ensure workflow_structure is explicitly set if needed by profile graph
        profile_initial_state["workflow_structure"] = workflow_structure
        profile_initial_state["messages"] = [
            {"role": "system", "content": "Generate user profiles based on the workflow structure."},
        ]
        # Reset fields specific to profile generation if necessary
        profile_initial_state["conversation_goals"] = []
        profile_initial_state["built_profiles"] = []

        # Run Profile Generation
        print("\n--- Generating user profiles ---")
        profile_thread_id = f"profile_analysis_{uuid.uuid4()}"
        profile_result = self._profile_graph.invoke(
            profile_initial_state,
            config={"configurable": {"thread_id": profile_thread_id}},
        )

        # Use 'conversation_goals' as the key holding the final profiles per profile_graph definition
        generated_profiles = profile_result.get("conversation_goals", [])
        print(f"--- Analysis complete, {len(generated_profiles)} profiles generated ---")

        return {
            "discovered_functionalities": workflow_structure,
            "built_profiles": generated_profiles,
        }
