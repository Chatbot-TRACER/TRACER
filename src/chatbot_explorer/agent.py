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
from chatbot_explorer.utils.logging_utils import get_logger
from connectors.chatbot_connectors import Chatbot

from .graphs.profile_graph import build_profile_generation_graph
from .graphs.structure_graph import build_structure_graph
from .schemas.graph_state_model import State

logger = get_logger()


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
        logger.debug("Initializing exploration process")
        # Initialize results storage
        conversation_sessions = []
        current_graph_state = self._initialize_graph_state()

        # Perform initial probing steps
        logger.verbose("Beginning initial chatbot probing steps")
        supported_languages = self._detect_languages(chatbot_connector)
        fallback_message = self._detect_fallback(chatbot_connector)

        # Create exploration parameters
        exploration_params: ExplorationParams = {
            "max_sessions": max_sessions,
            "max_turns": max_turns,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

        logger.debug("Exploration parameters prepared: %d sessions, %d turns per session", max_sessions, max_turns)

        # Run the main exploration sessions
        conversation_sessions, current_graph_state = self._run_exploration_sessions(
            chatbot_connector, exploration_params, current_graph_state
        )

        # Convert final root nodes to dictionaries for the result
        functionality_dicts = [node.to_dict() for node in current_graph_state["root_nodes"]]
        logger.debug("Converted %d root nodes to dictionary format", len(functionality_dicts))

        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

    def _initialize_graph_state(self) -> ExplorationGraphState:
        """Initialize the graph state for exploration."""
        logger.debug("Initializing exploration graph state")
        return {
            "root_nodes": [],
            "pending_nodes": [],
            "explored_nodes": set(),
        }

    def _detect_languages(self, chatbot_connector: Chatbot) -> list[str]:
        """Detect languages supported by the chatbot."""
        logger.verbose("\nProbing Chatbot Language")
        initial_probe_query = "Hello"
        logger.debug("Sending initial language probe message: '%s'", initial_probe_query)

        is_ok, probe_response = chatbot_connector.execute_with_input(initial_probe_query)
        supported_languages = ["English"]  # Default

        if is_ok and probe_response:
            logger.verbose("Initial response received: '%s...'", probe_response[:30])
            try:
                logger.debug("Analyzing response to detect supported languages")
                detected_langs = extract_supported_languages(probe_response, self.llm)
                if detected_langs:
                    supported_languages = detected_langs
                    logger.info("\nDetected initial language(s): %s", supported_languages)
                else:
                    logger.warning("Could not detect language from initial probe, defaulting to English")
            except (ValueError, TypeError):
                logger.exception("Error during initial language detection. Defaulting to English")
        else:
            logger.error("Could not get initial response from chatbot for language probe. Defaulting to English")

        return supported_languages

    def _detect_fallback(self, chatbot_connector: Chatbot) -> str | None:
        """Detect the chatbot's fallback message."""
        logger.verbose("\nDetecting chatbot fallback message")
        fallback_message = None

        try:
            logger.debug("Executing fallback detection sequence")
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
            if fallback_message:
                fallback_preview_length = 30
                logger.info('\nDetected fallback message: "%s..."', fallback_message[:fallback_preview_length])
            else:
                logger.warning("Could not detect a fallback message")
        except (ValueError, KeyError, AttributeError):
            logger.exception("Error during fallback detection. Proceeding without fallback")

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
        logger.info("\n=== Beginning Exploration Sessions ===")

        while session_num < params["max_sessions"]:
            # Determine which node to explore next
            explore_node, session_type = self._select_next_node(current_graph_state, session_num)

            # Skip already explored nodes
            if explore_node and explore_node.name in current_graph_state["explored_nodes"]:
                logger.verbose("Skipping already explored node: '%s'", explore_node.name)
                session_num += 1
                continue

            # Configure and run a single session
            session_params: SessionParams = {
                "session_num": session_num,
                "max_sessions": params["max_sessions"],
                "max_turns": params["max_turns"],
            }

            logger.debug("Configuring session %d with %d maximum turns", session_num + 1, params["max_turns"])
            session_config = self._create_session_config(
                session_params,
                chatbot_connector,
                params,
                explore_node,
                current_graph_state,
            )

            logger.debug("Running exploration session %d", session_num + 1)
            conversation_history, updated_graph_state = run_exploration_session(
                config=session_config,
            )

            conversation_sessions.append(conversation_history)
            logger.debug(
                "Session %d completed, conversation history captured (%d turns)",
                session_num + 1,
                len(conversation_history),
            )

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
            logger.debug("Selected node '%s' from pending queue", explore_node.name)
        elif session_num > 0:
            logger.verbose("Pending nodes queue is empty. Performing general exploration")

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
        logger.debug("Creating session configuration for session %d", session_params["session_num"] + 1)
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
        logger.info("\n=== Completed %d exploration sessions ===", session_count)

        if graph_state["pending_nodes"]:
            logger.verbose("NOTE: %d nodes still remain in the pending queue", len(graph_state["pending_nodes"]))
        else:
            logger.verbose("All discovered nodes were explored")

        logger.info("Discovered %d root functionalities after exploration", len(graph_state["root_nodes"]))

    def run_analysis(self, exploration_results: dict[str, Any]) -> dict[str, list[Any]]:
        """Runs the LangGraph analysis pipeline using pre-compiled graphs.

        Args:
            exploration_results: A dictionary containing results from the exploration phase.

        Returns:
            A dictionary containing 'discovered_functionalities' and 'built_profiles'.
        """
        logger.info("\n--- Preparing for analysis phase ---")
        logger.debug(
            "Initializing structure analysis with %d conversation sessions",
            len(exploration_results.get("conversation_sessions", [])),
        )

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
        logger.info("\n--- Running workflow structure inference ---")
        logger.verbose("Creating analysis thread for structure inference")
        structure_thread_id = f"structure_analysis_{uuid.uuid4()}"

        logger.debug("Starting structure graph invocation")
        structure_result = self._structure_graph.invoke(
            structure_initial_state,
            config={"configurable": {"thread_id": structure_thread_id}},
        )
        workflow_structure = structure_result.get("discovered_functionalities", {})
        logger.info("--- Structure inference complete ---")
        logger.verbose("Structure inference discovered %d workflow nodes", len(workflow_structure))

        # 2. Prepare initial state for the profile graph
        logger.debug("Preparing state for profile generation")
        profile_initial_state = structure_result.copy()
        profile_initial_state["workflow_structure"] = workflow_structure
        profile_initial_state["messages"] = [
            {"role": "system", "content": "Generate user profiles based on the workflow structure."},
        ]
        profile_initial_state["conversation_goals"] = []
        profile_initial_state["built_profiles"] = []

        # -- Run Profile Generation --
        logger.info("\n--- Generating user profiles ---")
        logger.verbose("Creating analysis thread for profile generation")
        profile_thread_id = f"profile_analysis_{uuid.uuid4()}"

        logger.debug("Starting profile graph invocation")
        profile_result = self._profile_graph.invoke(
            profile_initial_state,
            config={"configurable": {"thread_id": profile_thread_id}},
        )

        generated_profiles = profile_result.get("built_profiles", [])
        logger.info("--- Analysis complete, %d profiles generated ---", len(generated_profiles))
        logger.debug("Profile generation finished successfully")

        return {
            "discovered_functionalities": workflow_structure,
            "built_profiles": generated_profiles,
        }
