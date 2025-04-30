from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from chatbot_explorer.session import run_exploration_session
from chatbot_explorer.conversation.fallback_detection import extract_fallback_message
from chatbot_explorer.conversation.language_detection import extract_supported_languages

from .nodes.conversation_params_node import conversation_params_node
from .nodes.goal_generator_node import goal_generator_node
from .nodes.profile_builder_node import profile_builder_node
from .nodes.profile_validator_node import profile_validator_node
from .nodes.structure_builder_node import structure_builder_node
from .schemas.state import State

from .graphs.structure_graph import build_structure_graph
from .graphs.profile_graph import build_profile_generation_graph

from typing import Any, Dict, List, Optional, Set
import uuid


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

    def run_exploration(self, chatbot_connector, max_sessions: int, max_turns: int) -> Dict[str, Any]:
        """
        Runs the initial probing and the main exploration loop.

        Args:
            chatbot_connector: An instance of a chatbot connector class.
            max_sessions (int): Maximum number of exploration sessions to run.
            max_turns (int): Maximum turns per exploration session.

        Returns:
            Dict[str, Any]: A dictionary containing exploration results:
                            - conversation_sessions: List of conversation histories.
                            - root_nodes_dict: List of root FunctionalityNode objects as dicts.
                            - supported_languages: List of detected languages.
                            - fallback_message: Detected fallback message string.
        """
        # Initialize results and state tracking
        conversation_sessions: List[List[Dict[str, str]]] = []
        supported_languages: List[str] = []
        fallback_message: Optional[str] = None
        root_nodes: List[FunctionalityNode] = []
        pending_nodes: List[FunctionalityNode] = []
        explored_nodes: Set[str] = set()

        # --- Initial Language Detection ---
        print("\n--- Probing Chatbot Language ---")
        initial_probe_query = "Hello"
        is_ok, probe_response = chatbot_connector.execute_with_input(initial_probe_query)
        if is_ok and probe_response:
            print(f"   Initial response received: '{probe_response[:60]}...'")
            try:
                detected_langs = extract_supported_languages(probe_response, self.llm)
                if detected_langs:
                    supported_languages = detected_langs
                    print(f"   Detected initial language(s): {supported_languages}")
                else:
                    print("   Could not detect language from initial probe, defaulting to English.")
            except Exception as lang_e:
                print(f"   Error during initial language detection: {lang_e}. Defaulting to English.")
        else:
            print("   Could not get initial response from chatbot for language probe. Defaulting to English.")
        if not supported_languages:
            supported_languages = ["English"]

        # --- Initial Fallback Message Detection ---
        print("\n--- Attempting to detect chatbot fallback message ---")
        try:
            # Use the agent's LLM instance directly
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
            if fallback_message:
                print(f'   Detected fallback message: "{fallback_message[:50]}..."')
            else:
                print("   Could not detect a fallback message.")
        except Exception as fb_e:
            print(f"   Error during fallback detection: {fb_e}. Proceeding without fallback.")
            fallback_message = None

        # --- Exploration Loop ---
        session_num = 0
        while session_num < max_sessions:
            current_session_index = session_num
            print(f"\n=== Starting Session {current_session_index + 1}/{max_sessions} ===")

            explore_node = None
            session_type_log = "General Exploration"

            if pending_nodes:
                explore_node = pending_nodes.pop(0)
                if explore_node.name in explored_nodes:
                    print(f"--- Skipping already explored node: '{explore_node.name}' ---")
                    session_num += 1
                    continue
                session_type_log = f"Exploring functionality '{explore_node.name}'"
            elif session_num > 0:
                print("   Pending nodes queue is empty. Performing general exploration.")

            print(f"   Session Type: {session_type_log}")

            # Execute one exploration session, passing self (the agent instance)
            (
                conversation_history,
                updated_roots,
                updated_pending,
                updated_explored,
            ) = run_exploration_session(
                session_num=current_session_index,
                max_sessions=max_sessions,
                max_turns=max_turns,
                llm=self.llm,
                the_chatbot=chatbot_connector,
                fallback_message=fallback_message,
                current_node=explore_node,
                explored_nodes=explored_nodes,
                pending_nodes=pending_nodes,
                root_nodes=root_nodes,
                supported_languages=supported_languages,
            )

            conversation_sessions.append(conversation_history)
            root_nodes = updated_roots
            pending_nodes = updated_pending
            explored_nodes = updated_explored
            session_num += 1

        # --- Post-Exploration Summary ---
        print(f"\n=== Completed {session_num} exploration sessions ===")
        if pending_nodes:
            print(f"   NOTE: {len(pending_nodes)} nodes still remain in the pending queue.")
        else:
            print("   All discovered nodes were explored.")
        print(f"Discovered {len(root_nodes)} root functionalities after exploration.")

        functionality_dicts = [node.to_dict() for node in root_nodes]

        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

    def run_analysis(self, exploration_results: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Runs the LangGraph analysis pipeline using pre-compiled graphs.

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
            discovered_functionalities=exploration_results.get("root_nodes_dict", {}),
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
        # Access the pre-compiled graph directly via self
        structure_result = self._structure_graph.invoke(
            structure_initial_state, config={"configurable": {"thread_id": structure_thread_id}}
        )
        workflow_structure = structure_result.get("discovered_functionalities", {})
        print("--- Structure inference complete ---")

        # 2. Prepare initial state for the profile graph
        # Start with the state resulting from structure analysis
        profile_initial_state = structure_result.copy()  # Creates a shallow copy
        profile_initial_state["workflow_structure"] = workflow_structure
        profile_initial_state["messages"] = [
            {"role": "system", "content": "Generate user profiles based on the workflow structure."},
        ]
        profile_initial_state["conversation_goals"] = []
        profile_initial_state["built_profiles"] = []

        # -- Run Profile Generation --
        print("\n--- Generating user profiles ---")
        profile_thread_id = f"profile_analysis_{uuid.uuid4()}"
        # Access the pre-compiled graph directly via self
        profile_result = self._profile_graph.invoke(
            profile_initial_state, config={"configurable": {"thread_id": profile_thread_id}}
        )

        generated_profiles = profile_result.get("built_profiles", [])
        print(f"--- Analysis complete, {len(generated_profiles)} profiles generated ---")

        return {
            "discovered_functionalities": workflow_structure,
            "built_profiles": generated_profiles,
        }
