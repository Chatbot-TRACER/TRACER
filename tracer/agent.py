"""ChatbotExplorer Agent, contains methods to run the exploration and analysis."""

import pprint
import uuid
from typing import Any, TypedDict

import requests
from google.api_core.exceptions import PermissionDenied
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from openai import AuthenticationError

from tracer.constants import MIN_NODES_FOR_DEDUPLICATION

try:
    from google.auth.exceptions import DefaultCredentialsError, RefreshError
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from chatbot_connectors import Chatbot

from tracer.analysis.functionality_refinement import (
    _process_node_group_for_merge,
    is_duplicate_functionality,
)
from tracer.conversation.fallback_detection import extract_fallback_message
from tracer.conversation.language_detection import extract_supported_languages
from tracer.conversation.session import (
    ExplorationGraphState,
    ExplorationSessionConfig,
    run_exploration_session,
)
from tracer.schemas.functionality_node_model import FunctionalityNode
from tracer.utils.logging_utils import get_logger
from tracer.utils.token_tracker_callback import TokenUsageTracker
from tracer.utils.tracer_error import LLMError

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
            model_name (str): The name of the model to use (OpenAI or Gemini).
        """
        self.model_name = model_name  # Store model name as instance variable
        self.token_tracker = TokenUsageTracker()  # Initialize token tracker
        self.llm = self._initialize_llm(model_name)
        self.memory = MemorySaver()

        self._structure_graph = build_structure_graph(self.llm, self.memory)
        self._profile_graph = build_profile_generation_graph(self.llm, self.memory)

    def _initialize_llm(self, model_name: str) -> BaseChatModel:
        """Initialize the language model based on the model name.

        Args:
            model_name (str): The name of the model to use.

        Returns:
            BaseChatModel: An instance of the appropriate LLM.

        Raises:
            ImportError: If trying to use Gemini/OpenAI models but the required package is not installed.
            LLMError: If authentication or API-related errors occur during initialization.
        """
        llm: BaseChatModel

        # Check if this is a Gemini model
        if model_name.lower().startswith("gemini"):
            if not GEMINI_AVAILABLE:
                gemini_error_msg = "To use Gemini models, please install the required package: pip install langchain-google-genai google-generativeai"
                logger.exception("Gemini support not available")
                raise ImportError(gemini_error_msg)

            logger.info("Initializing Gemini model: %s", model_name)

            try:
                llm = ChatGoogleGenerativeAI(model=model_name, callbacks=[self.token_tracker])
            except (DefaultCredentialsError, RefreshError) as e:
                logger.exception("Gemini authentication failed. Please check your API key or credentials.")
                msg = "Gemini authentication failed."
                raise LLMError(msg) from e
            except Exception as e:
                logger.exception("Failed to initialize Gemini model")
                msg = f"Failed to initialize Gemini model '{model_name}'."
                raise LLMError(msg) from e

            try:
                logger.debug("Performing health check on Gemini model...")
                llm.invoke("test")
            except PermissionDenied as e:
                logger.exception("Gemini API key is invalid or lacks permissions. Please check your credentials.")
                msg = "Gemini authentication failed."
                raise LLMError(msg) from e
            except Exception as e:
                logger.exception("Failed to verify connection to Gemini")
                msg = "Gemini health check failed."
                raise LLMError(msg) from e

        # Handle OpenAI models
        else:
            try:
                logger.info("Initializing OpenAI model: %s", model_name)
                llm = ChatOpenAI(model=model_name, callbacks=[self.token_tracker], request_timeout=60)

                logger.debug("Performing health check on OpenAI model...")
                llm.invoke("test")
            except AuthenticationError as e:
                logger.exception("OpenAI API key is invalid or has expired. Please check your credentials.")
                msg = "OpenAI authentication failed."
                raise LLMError(msg) from e
            except ImportError as e:
                openai_error_msg = (
                    "To use OpenAI models, please install the required package: pip install langchain-openai openai"
                )
                logger.exception("OpenAI support not available")
                raise ImportError(openai_error_msg) from e
            except Exception as e:
                error_msg = (
                    f"Error initializing OpenAI model '{model_name}'. Please check your OpenAI API key and model name."
                )
                logger.exception("OpenAI initialization failed")
                raise LLMError(error_msg) from e

        return llm

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

        try:
            chatbot_connector.health_check()
        except requests.RequestException:
            logger.exception("Initial health check failed")
            raise  # Re-raise to be caught by the main error handler

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
        logger.debug(
            "Converted %d root nodes to dictionary format (root_nodes_dict). Checking for parameter options:",
            len(functionality_dicts),
        )
        for i, node_dict in enumerate(functionality_dicts):
            logger.debug("  root_nodes_dict[%d]:\n%s", i, pprint.pformat(node_dict, indent=2, width=120))

        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
            "token_usage": self.token_tracker.get_summary(),  # Add token usage to results
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

        try:
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

        except requests.RequestException:
            logger.exception("Connection error during language detection")
            raise  # Re-raise to be caught by the main error handler

        return supported_languages

    def _detect_fallback(self, chatbot_connector: Chatbot) -> str | None:
        """Detect the fallback message of the chatbot."""
        logger.verbose("\nProbing Chatbot Fallback Response")
        try:
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
        except requests.RequestException:
            logger.exception("Connection error during fallback detection")
            raise  # Re-raise to be caught by the main error handler
        except (ValueError, TypeError):
            logger.exception("Could not determine fallback message due to an unexpected error")
            fallback_message = None

        if fallback_message:
            logger.info("Detected fallback message: '%s...'", fallback_message[:30])
        else:
            logger.warning("Could not detect a fallback message for the chatbot.")

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

            logger.debug("Current graph state updated after session %d:", session_num)
            logger.debug("  Root nodes (%d):", len(updated_graph_state["root_nodes"]))
            for root_node in updated_graph_state["root_nodes"]:
                for line in root_node.to_detailed_string(indent_level=0).splitlines():
                    logger.debug("    %s", line)
            logger.debug("  Pending nodes (%d):", len(updated_graph_state["pending_nodes"]))
            for node in updated_graph_state["pending_nodes"]:
                logger.debug("    - %s", node)
            logger.debug(
                "  Explored nodes (%d): %s",
                len(updated_graph_state["explored_nodes"]),
                sorted(updated_graph_state["explored_nodes"]),
            )

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

        logger.info("\nDiscovered %d root functionalities after exploration", len(graph_state["root_nodes"]))

    def _aggressive_node_deduplication(self, nodes: list[FunctionalityNode]) -> list[FunctionalityNode]:
        """Aggressively deduplicate nodes using pairwise comparisons.

        Instead of relying on name-based grouping, this method compares each node
        against every other node and merges them if they are deemed similar.

        Args:
            nodes: List of functionality nodes to deduplicate

        Returns:
            List of deduplicated functionality nodes
        """
        if not nodes or len(nodes) < MIN_NODES_FOR_DEDUPLICATION:
            return nodes

        logger.verbose("Starting aggressive deduplication of %d nodes", len(nodes))

        # Make a copy of the nodes to work with
        remaining_nodes = nodes.copy()

        # Keep track of merged node pairs to avoid checking them again
        merged_pairs = set()

        # Flag to track if any merges happened in this pass
        any_merges_happened = True

        # Continue until no more merges happen
        while any_merges_happened and len(remaining_nodes) > 1:
            any_merges_happened = False
            new_remaining_nodes = []

            # Process all remaining nodes
            while remaining_nodes:
                current_node = remaining_nodes.pop(0)

                # Compare with each other node
                i = 0
                while i < len(remaining_nodes):
                    other_node = remaining_nodes[i]

                    # Skip if we've already tried to merge this pair
                    pair_key = frozenset([current_node.name, other_node.name])
                    if pair_key in merged_pairs:
                        i += 1
                        continue

                    # Check if these nodes are duplicates
                    is_dup, _ = is_duplicate_functionality(current_node, [other_node], self.llm)

                    if is_dup:
                        logger.debug(
                            "Found duplicate nodes: '%s' and '%s'. Attempting to merge.",
                            current_node.name,
                            other_node.name,
                        )

                        # Try to merge the nodes
                        merge_candidates = [current_node, other_node]
                        merged_result = _process_node_group_for_merge(merge_candidates, self.llm)

                        if len(merged_result) == 1:
                            # Successful merge
                            logger.debug(
                                "Successfully merged '%s' and '%s' into '%s'",
                                current_node.name,
                                other_node.name,
                                merged_result[0].name,
                            )

                            # Replace current_node with the merged node
                            current_node = merged_result[0]

                            # Remove the other node from consideration
                            remaining_nodes.pop(i)

                            any_merges_happened = True
                        else:
                            # Mark this pair as processed
                            merged_pairs.add(pair_key)
                            i += 1
                    else:
                        i += 1

                # Add the current node back to the list (possibly merged)
                new_remaining_nodes.append(current_node)

            # Update the remaining nodes for the next iteration
            remaining_nodes = new_remaining_nodes

            if any_merges_happened:
                logger.debug(
                    "Completed a merge pass, %d nodes remaining",
                    len(remaining_nodes),
                )

        logger.verbose(
            "Aggressive deduplication complete: reduced from %d to %d nodes", len(nodes), len(remaining_nodes)
        )
        return remaining_nodes

    def run_analysis(
        self, exploration_results: dict[str, Any], *, nested_forward: bool = False, profile_model: str | None = None
    ) -> dict[str, list[Any]]:
        """Runs the LangGraph analysis pipeline using pre-compiled graphs.

        Args:
            exploration_results: Results from the exploration phase
            nested_forward: Whether to use nested forward() chaining in variable definitions
            profile_model: Model to use for profile generation (defaults to exploration model)

        Returns:
            Results from the analysis phase
        """
        # Use profile_model if provided, otherwise fall back to exploration model
        model_for_profiles = profile_model or self.model_name

        conversation_count = len(exploration_results.get("conversation_sessions", []))
        functionality_count = len(exploration_results.get("root_nodes_dict", {}))
        logger.debug(
            "Initializing analysis with %d conversation sessions and %d discovered functionalities",
            conversation_count,
            functionality_count,
        )
        logger.debug("Using model '%s' for exploration and '%s' for profiles", self.model_name, model_for_profiles)

        # Deduplicate functionalities before analysis
        root_nodes_dict = exploration_results.get("root_nodes_dict", {})
        if root_nodes_dict:
            logger.info("\nDeduplicating functionalities before analysis")
            # Convert dictionaries back to FunctionalityNode objects for deduplication
            functionality_nodes = []
            for node_dict in root_nodes_dict:
                node = FunctionalityNode.from_dict(node_dict)
                functionality_nodes.append(node)

            logger.debug("--- Functionality Nodes Before Deduplication ---")
            if not functionality_nodes:
                logger.debug("No functionality nodes to display before deduplication.")
            for i, node in enumerate(functionality_nodes):
                logger.debug("Node %d (Before Deduplication):", i + 1)
                for line in node.to_detailed_string(indent_level=1).splitlines():
                    logger.debug("  %s", line)

            # Apply aggressive pairwise deduplication instead of name-based grouping
            logger.debug(
                "Running aggressive pairwise deduplication on %d functionality nodes", len(functionality_nodes)
            )
            deduplicated_nodes = self._aggressive_node_deduplication(functionality_nodes)
            logger.info(
                "Deduplication complete: %d nodes reduced to %d nodes",
                len(functionality_nodes),
                len(deduplicated_nodes),
            )

            logger.debug("--- Functionality Nodes After Deduplication ---")
            for i, node in enumerate(deduplicated_nodes):
                logger.debug("Node %d (After Deduplication):", i + 1)
                for line in node.to_detailed_string(indent_level=1).splitlines():
                    logger.debug("  %s", line)

            # Convert back to dictionaries for the workflow builder
            root_nodes_dict = [node.to_dict() for node in deduplicated_nodes]
            exploration_results["root_nodes_dict"] = root_nodes_dict
            logger.debug("Updated exploration results with %d deduplicated nodes", len(root_nodes_dict))

        # 1. Structure analysis phase
        logger.info("\nStep 1: Workflow structure inference")
        logger.info("--------------------------\n")

        # Prepare initial state for the structure graph
        structure_initial_state = State(
            messages=[{"role": "system", "content": "Infer structure from conversation history."}],
            conversation_history=exploration_results.get("conversation_sessions", []),
            discovered_functionalities=exploration_results.get("root_nodes_dict", {}),
            built_profiles=[],
            discovered_limitations=[],
            current_session=conversation_count,
            exploration_finished=True,
            conversation_goals=[],
            supported_languages=exploration_results.get("supported_languages", []),
            fallback_message=exploration_results.get("fallback_message", ""),
            workflow_structure=None,
            nested_forward=nested_forward,
            model=self.model_name,
        )

        # Run Structure Inference
        logger.debug("Creating analysis thread for structure inference")
        structure_thread_id = f"structure_analysis_{uuid.uuid4()}"

        structure_result = self._structure_graph.invoke(
            structure_initial_state,
            config={"configurable": {"thread_id": structure_thread_id}},
        )
        workflow_structure = structure_result.get("discovered_functionalities", {})
        logger.debug(
            "Workflow structure generated by LLM (%d top-level nodes). Checking for parameter options:",
            len(workflow_structure),
        )
        logger.debug("  Workflow Structure:\n%s", pprint.pformat(workflow_structure, indent=2, width=120))

        # 2. Profile generation phase
        logger.info("\nStep 2: User profile generation")
        logger.info("--------------------------\n")

        # Prepare initial state for profile generation
        profile_initial_state = structure_result.copy()
        profile_initial_state["workflow_structure"] = workflow_structure
        profile_initial_state["messages"] = [
            {"role": "system", "content": "Generate user profiles based on the workflow structure."},
        ]
        profile_initial_state["conversation_goals"] = []
        profile_initial_state["built_profiles"] = []
        profile_initial_state["model"] = model_for_profiles

        # Run Profile Generation
        logger.debug("Creating analysis thread for profile generation")
        profile_thread_id = f"profile_analysis_{uuid.uuid4()}"

        profile_result = self._profile_graph.invoke(
            profile_initial_state,
            config={"configurable": {"thread_id": profile_thread_id}},
        )

        generated_profiles = profile_result.get("built_profiles", [])

        return {
            "discovered_functionalities": workflow_structure,
            "built_profiles": generated_profiles,
            "token_usage": self.token_tracker.get_summary(),  # Add token usage to results
        }
