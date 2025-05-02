"""Manages the conversational exploration sessions with the target chatbot."""

import enum
import secrets
from typing import TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chatbot_explorer.analysis.functionality_extraction import extract_functionality_nodes
from chatbot_explorer.analysis.functionality_refinement import (
    is_duplicate_functionality,
    merge_similar_functionalities,
    validate_parent_child_relationship,
)
from chatbot_explorer.prompts.session_prompts import (
    get_explorer_system_prompt,
    get_force_topic_change_instruction,
    get_initial_question_prompt,
    get_language_instruction,
    get_rephrase_prompt,
    get_session_focus,
    get_translation_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode
from chatbot_explorer.utils.logging_utils import get_logger
from connectors.chatbot_connectors import Chatbot

from .conversation_utils import _get_all_nodes
from .fallback_detection import (
    is_semantically_fallback,
)

logger = get_logger()

MAX_LOG_MESSAGE_LENGTH = 255


class ExplorationGraphState(TypedDict):
    """Represents the state of the functionality graph being explored.

    Attributes:
        root_nodes: List of root nodes in the graph.
        pending_nodes: List of nodes that are pending exploration.
        explored_nodes: Set of node identifiers that have been explored.
    """

    root_nodes: list[FunctionalityNode]
    pending_nodes: list[FunctionalityNode]
    explored_nodes: set[str]


class ConversationContext(TypedDict):
    """Contextual information needed during the conversation loop."""

    llm: BaseLanguageModel
    the_chatbot: Chatbot
    fallback_message: str | None


class ExplorationSessionConfig(TypedDict):
    """Configuration and state required to run a single exploration session.

    Attributes:
        session_num: The current session number.
        max_sessions: The maximum number of sessions.
        max_turns: The maximum number of turns per session.
        llm: The language model instance.
        the_chatbot: The chatbot instance.
        fallback_message: The fallback message for the chatbot.
        current_node: The current functionality node being explored.
        graph_state: The current state of the exploration graph.
        supported_languages: List of languages supported in the session.



    """

    session_num: int
    max_sessions: int
    max_turns: int
    llm: BaseLanguageModel
    the_chatbot: Chatbot
    fallback_message: str | None
    current_node: FunctionalityNode | None
    graph_state: ExplorationGraphState
    supported_languages: list[str]


class InteractionOutcome(enum.Enum):
    """Represents the possible outcomes of an interaction with the chatbot."""

    SUCCESS = 0
    COMM_ERROR = 1
    FALLBACK_DETECTED = 2
    RETRY_FAILED = 3


def _generate_initial_question(
    current_node: FunctionalityNode | None, primary_language: str, llm: BaseLanguageModel
) -> str:
    """Generates the initial question for the exploration session."""
    lang_lower = primary_language.lower()
    if current_node:
        # Ask about the specific node
        question_prompt = get_initial_question_prompt(current_node, primary_language)
        question_response = llm.invoke(question_prompt)
        initial_question = question_response.content.strip().strip("\"'")
    else:
        # Ask a general question for the first session
        possible_greetings = [
            "Hello! What can you help me with today?",
            "Hello, how can I get started?",
            "I'm interested in using your services. What's available",
            "Can you list your main functions or services?",
        ]
        greeting_en = secrets.choice(possible_greetings)

        # Translate if needed
        if lang_lower != "english":
            try:
                translation_prompt = get_translation_prompt(greeting_en, primary_language)
                translated_greeting = llm.invoke(translation_prompt).content.strip().strip("\"'")
                # Check if translation looks okay
                if translated_greeting and len(translated_greeting.split()) > 1:
                    initial_question = translated_greeting
                else:  # Use English if translation failed
                    initial_question = greeting_en
            except ValueError:
                logger.warning("Failed to translate initial greeting to %s. Using English instead.", primary_language)
                initial_question = greeting_en  # Fallback
        else:
            initial_question = greeting_en  # Use English

    return initial_question


def _add_new_node_to_graph(
    node: FunctionalityNode,
    current_node: FunctionalityNode | None,
    graph_state: ExplorationGraphState,
    llm: BaseLanguageModel,
) -> bool:
    """Attempts to add a new node to the graph state, handling parent relationships.

    Returns True if the node was added (either as root or child), False otherwise.
    """
    is_added_as_child = False
    # Try adding as a child if exploring a specific node
    if current_node:
        relationship_valid = validate_parent_child_relationship(current_node, node, llm)
        if relationship_valid:
            current_node.add_child(node)
            logger.debug("Added '%s' (child of '%s')", node.name, current_node.name)
            is_added_as_child = True

    # If not added as child (either no current_node or relationship invalid), add as root
    if not is_added_as_child:
        logger.debug("Added '%s' (new root/standalone functionality)", node.name)
        graph_state["root_nodes"].append(node)

    # Add to pending queue if it hasn't been explored yet
    if node.name not in graph_state["explored_nodes"]:
        graph_state["pending_nodes"].append(node)

    return True  # Indicate node was added


def _analyze_session_and_update_nodes(
    conversation_history_lc: list[BaseMessage],
    llm: BaseLanguageModel,
    current_node: FunctionalityNode | None,
    graph_state: ExplorationGraphState,
) -> tuple[list[dict[str, str]], ExplorationGraphState]:
    """Analyzes conversation, extracts functionalities, and updates the exploration graph state.

    Args:
        conversation_history_lc: The conversation history using LangChain message objects.
        llm: The language model instance.
        current_node: The functionality node that was the focus of this session, if any.
        graph_state: The current state of the exploration graph (roots, pending, explored).

    Returns:
        A tuple containing the conversation history as a list of dictionaries and the
        updated exploration graph state.
    """
    # Convert LangChain messages to simple dicts for analysis functions compatibility
    conversation_history_dict = [
        {
            "role": "system" if isinstance(m, SystemMessage) else ("assistant" if isinstance(m, AIMessage) else "user"),
            "content": m.content,
        }
        for m in conversation_history_lc
    ]

    # Extract potential new functionalities mentioned in this session's conversation
    logger.verbose("Analyzing conversation for new functionalities...")
    newly_extracted_nodes = extract_functionality_nodes(conversation_history_dict, llm, current_node)

    # Track nodes added in this session for the summary
    session_added_nodes = []

    # Process the extracted nodes if any were found
    if newly_extracted_nodes:
        logger.debug("Discovered %d potential new functionality nodes", len(newly_extracted_nodes))

        # Merge similar nodes discovered *within this session* before further processing
        processed_new_nodes = merge_similar_functionalities(newly_extracted_nodes, llm)

        # Get a snapshot of all nodes currently in the graph for duplicate checking
        # This is done once before processing the new batch.
        nodes_in_graph_before_adding = []
        for root in graph_state["root_nodes"]:
            nodes_in_graph_before_adding.extend(_get_all_nodes(root))

        # Keep track of nodes added *during this analysis step* for subsequent duplicate checks
        nodes_added_this_analysis = []

        for node in processed_new_nodes:
            # Check for duplicates against nodes already in the graph AND nodes just added in this analysis step
            combined_check_list = nodes_in_graph_before_adding + nodes_added_this_analysis
            if not is_duplicate_functionality(node, combined_check_list, llm):
                # Attempt to add the non-duplicate node to the graph
                was_added = _add_new_node_to_graph(node, current_node, graph_state, llm)
                if was_added:
                    # If added successfully, include it in the list for subsequent duplicate checks within this loop
                    nodes_added_this_analysis.append(node)
                    session_added_nodes.append(node)
            else:
                logger.debug("Skipped duplicate functionality: '%s'", node.name)

        # After adding all new nodes from this session, re-run merge on the root nodes
        # This handles cases where a new root might be similar to an existing root.
        if graph_state["root_nodes"]:
            logger.debug("Merging root nodes after additions...")
            graph_state["root_nodes"] = merge_similar_functionalities(graph_state["root_nodes"], llm)

    # Mark the node focused on in this session as explored (if applicable)
    if current_node:
        graph_state["explored_nodes"].add(current_node.name)

    # Log summary of discoveries in this session
    if session_added_nodes:
        logger.info("Discovered %d new functionalities in this session:", len(session_added_nodes))
        for node in session_added_nodes:
            parent = current_node.name if current_node and node in current_node.children else "root"
            relationship = f"(child of '{parent}')" if parent != "root" else "(root functionality)"
            logger.info(" • %s %s", node.name, relationship)
    else:
        logger.info("No new functionalities discovered in this session")

    # Return the history dict and the updated graph state
    return conversation_history_dict, graph_state


def _get_explorer_next_message(
    conversation_history_lc: list[BaseMessage],
    llm: BaseLanguageModel,
    force_topic_change_instruction: str | None,
) -> str | None:
    """Gets the next message from the explorer LLM."""
    try:
        max_history_turns_for_llm = 10
        # Include system prompt + recent turns
        messages_for_llm = [conversation_history_lc[0]] + conversation_history_lc[-(max_history_turns_for_llm * 2) :]
        # Add force topic change instruction if applicable
        messages_for_llm_this_turn = (
            [*messages_for_llm, SystemMessage(content=force_topic_change_instruction)]
            if force_topic_change_instruction
            else messages_for_llm
        )
        logger.debug("Getting next explorer message from LLM")
        llm_response = llm.invoke(messages_for_llm_this_turn)
        return llm_response.content.strip()
    except (ValueError, RuntimeError):
        logger.exception("Error getting response from Explorer AI LLM. Ending session.")
        return None


def _handle_chatbot_interaction(
    explorer_message: str,
    the_chatbot: Chatbot,
    llm: BaseLanguageModel,
    fallback_message: str | None,
) -> tuple[InteractionOutcome, str]:
    """Interacts with the chatbot, handles retries on fallback/error, and returns outcome."""
    logger.debug("Sending message to chatbot")
    is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_message)

    if not is_ok:
        return InteractionOutcome.COMM_ERROR, "[Chatbot communication error]"

    original_chatbot_message = chatbot_message
    is_initial_fallback = fallback_message and is_semantically_fallback(chatbot_message, fallback_message, llm)
    is_initial_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message
    needs_retry = is_initial_fallback or is_initial_parsing_error

    if not needs_retry:
        return InteractionOutcome.SUCCESS, chatbot_message

    # --- Retry Logic ---
    failure_reason = "Fallback message" if is_initial_fallback else "Potential chatbot error (OUTPUT_PARSING_FAILURE)"
    logger.verbose("(%s detected. Rephrasing and retrying...)", failure_reason)
    rephrase_prompt = get_rephrase_prompt(explorer_message)
    rephrased_message_or_original = explorer_message  # Default to original
    try:
        rephrased_response = llm.invoke(rephrase_prompt)
        rephrased_message = rephrased_response.content.strip().strip("\"'")
        if rephrased_message and rephrased_message != explorer_message:
            logger.debug("Rephrased original message")
            rephrased_message_or_original = rephrased_message
        else:
            logger.debug("Failed to generate a different rephrasing. Retrying with original.")
    except (ValueError, RuntimeError):
        logger.exception("Error rephrasing message. Retrying with original.")

    logger.debug("Sending retry message to chatbot")
    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(rephrased_message_or_original)

    if is_ok_retry:
        is_retry_fallback = fallback_message and is_semantically_fallback(chatbot_message_retry, fallback_message, llm)
        is_retry_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message_retry
        if chatbot_message_retry != original_chatbot_message and not is_retry_fallback and not is_retry_parsing_error:
            logger.verbose("Retry successful!")
            return InteractionOutcome.SUCCESS, chatbot_message_retry  # Success after retry

        logger.verbose("Retry failed (still received fallback/error or same message)")
        # Return original message but indicate retry failed
        return InteractionOutcome.RETRY_FAILED, original_chatbot_message

    logger.verbose("Retry failed (communication error)")
    # Return original message but indicate retry failed (due to comms)
    return InteractionOutcome.RETRY_FAILED, original_chatbot_message


def _update_loop_state_after_interaction(
    outcome: InteractionOutcome,
    current_consecutive_failures: int,
) -> tuple[int, bool]:
    """Updates state based on interaction outcome enum."""
    new_consecutive_failures = current_consecutive_failures
    new_force_topic_change_next_turn = False

    if outcome == InteractionOutcome.COMM_ERROR:
        new_consecutive_failures += 1
        new_force_topic_change_next_turn = True
        logger.warning("Communication failure. Consecutive failures: %d", new_consecutive_failures)
    elif outcome == InteractionOutcome.FALLBACK_DETECTED:
        new_consecutive_failures += 1
        logger.debug("Initial fallback/error detected. Consecutive failures: %d", new_consecutive_failures)
    elif outcome == InteractionOutcome.RETRY_FAILED:
        new_consecutive_failures += 1
        new_force_topic_change_next_turn = True  # Force change after failed retry
        logger.debug("Retry failed. Consecutive failures: %d", new_consecutive_failures)
    elif outcome == InteractionOutcome.SUCCESS:
        if new_consecutive_failures > 0:
            logger.debug("Successful response. Resetting consecutive failures from %d", new_consecutive_failures)
        new_consecutive_failures = 0
        new_force_topic_change_next_turn = False

    return new_consecutive_failures, new_force_topic_change_next_turn


def _run_conversation_loop(
    max_turns: int,
    context: ConversationContext,
    initial_history: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """Runs the main conversation loop for a single session."""
    conversation_history_lc = initial_history[:]  # Work on a copy
    consecutive_failures = 0
    force_topic_change_next_turn = False
    turn_count = 1  # Start at 1 because the initial exchange is already in history

    # Extract context components for easier access
    llm = context["llm"]
    the_chatbot = context["the_chatbot"]
    fallback_message = context["fallback_message"]

    while True:
        # 1. Check Stop Conditions
        if turn_count >= max_turns:
            logger.debug("Reached maximum turns (%d). Ending session.", max_turns)
            break

        # 2. Determine if Topic Change is Needed
        force_topic_change_instruction = get_force_topic_change_instruction(
            consecutive_failures=consecutive_failures,
            force_topic_change_next_turn=force_topic_change_next_turn,
        )
        if force_topic_change_instruction:
            logger.verbose("Forcing topic change")
            if force_topic_change_next_turn:
                force_topic_change_next_turn = False

        # 3. Get Explorer's Next Message
        explorer_response_content = _get_explorer_next_message(
            conversation_history_lc, llm, force_topic_change_instruction
        )
        if explorer_response_content is None:
            break  # Error message logged inside helper
        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            logger.debug("Explorer has finished session exploration")
        logger.verbose(
            "Explorer: %s",
            explorer_response_content[:MAX_LOG_MESSAGE_LENGTH]
            + ("..." if len(explorer_response_content) > MAX_LOG_MESSAGE_LENGTH else ""),
        )
        conversation_history_lc.append(AIMessage(content=explorer_response_content))

        # 4. Interact with Chatbot (with retry logic)
        outcome, final_chatbot_message = _handle_chatbot_interaction(
            explorer_response_content, the_chatbot, llm, fallback_message
        )

        # 5. Update History
        conversation_history_lc.append(HumanMessage(content=final_chatbot_message))
        logger.verbose(
            "Chatbot: %s",
            final_chatbot_message[:MAX_LOG_MESSAGE_LENGTH]
            + ("..." if len(final_chatbot_message) > MAX_LOG_MESSAGE_LENGTH else ""),
        )

        # 6. Update Loop State (Failures, Topic Change Flag) using helper
        consecutive_failures, force_topic_change_next_turn = _update_loop_state_after_interaction(
            outcome=outcome,  # Pass the outcome enum
            current_consecutive_failures=consecutive_failures,
        )

        # Break loop immediately if communication failed
        if outcome == InteractionOutcome.COMM_ERROR:
            # Error message logged inside _update_loop_state_after_interaction
            break

        # 7. Check if Chatbot Ended Conversation
        if final_chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
            logger.debug("Chatbot ended the conversation. Ending session.")
            break

        # 8. Increment Turn Count
        turn_count += 1

    return conversation_history_lc, turn_count


def run_exploration_session(
    config: ExplorationSessionConfig,
) -> tuple[list[dict[str, str]], ExplorationGraphState]:
    """Runs one chat session to explore the chatbot's capabilities based on the provided configuration.

    Args:
        config: An ExplorationSessionConfig dictionary containing all necessary
            parameters and state for the session, including session number, turn limits,
            LLM instance, chatbot connector, target functionality node (if any),
            the current graph state, and supported languages.

    Returns:
        A tuple containing:
            - list[dict[str, str]]: The conversation history of the session, formatted
              as a list of dictionaries, where each dictionary has 'role' (system,
              assistant, or user) and 'content' keys.
            - ExplorationGraphState: The updated state of the exploration graph after
              analyzing the session's conversation and potentially adding or modifying
              functionality nodes.
    """
    # Extract parameters from the config object
    session_num = config["session_num"]
    max_sessions = config["max_sessions"]
    max_turns = config["max_turns"]
    llm = config["llm"]
    the_chatbot = config["the_chatbot"]
    fallback_message = config["fallback_message"]
    current_node = config["current_node"]
    graph_state = config["graph_state"]
    supported_languages = config["supported_languages"]

    # Log clear session start marker with basic info
    if current_node:
        logger.info(
            "\n=== Starting Exploration Session %d/%d: Exploring '%s' ===",
            session_num + 1,
            max_sessions,
            current_node.name,
        )
    else:
        logger.info("\n=== Starting Exploration Session %d/%d: General Exploration ===", session_num + 1, max_sessions)

    # --- Setup Session ---
    session_focus = get_session_focus(current_node)
    primary_language = supported_languages[0] if supported_languages else "English"
    language_instruction = get_language_instruction(supported_languages, primary_language)
    exploration_prompt = get_explorer_system_prompt(session_focus, language_instruction, max_turns)
    conversation_history_lc: list[BaseMessage] = [SystemMessage(content=exploration_prompt)]

    # --- Initial Exchange ---
    initial_question = _generate_initial_question(current_node, primary_language, llm)
    logger.debug("Initial question: %s", initial_question)

    # Log the explorer's first message for better visibility
    logger.verbose("Explorer: %s", initial_question)

    logger.debug("Sending initial question to chatbot")
    is_ok, chatbot_message = the_chatbot.execute_with_input(initial_question)

    turn_count = 0
    if not is_ok:
        logger.error("Error communicating with chatbot on initial message. Ending session.")
        conversation_history_lc.append(AIMessage(content=initial_question))
        conversation_history_lc.append(HumanMessage(content="[Chatbot communication error on initial message]"))
    else:
        # Log the chatbot's first response for better visibility
        logger.verbose(
            "Chatbot: %s",
            chatbot_message[:MAX_LOG_MESSAGE_LENGTH] + ("..." if len(chatbot_message) > MAX_LOG_MESSAGE_LENGTH else ""),
        )

        conversation_history_lc.append(AIMessage(content=initial_question))
        conversation_history_lc.append(HumanMessage(content=chatbot_message))
        turn_count = 1

        # --- Conversation Loop ---
        if turn_count < max_turns:
            # Create context dictionary for the loop
            loop_context: ConversationContext = {
                "llm": llm,
                "the_chatbot": the_chatbot,
                "fallback_message": fallback_message,
            }
            conversation_history_lc, turn_count = _run_conversation_loop(
                max_turns=max_turns,
                context=loop_context,  # Pass context object
                initial_history=conversation_history_lc,
            )

    # --- Post-Session Analysis ---
    conversation_history_dict, updated_graph_state = _analyze_session_and_update_nodes(
        conversation_history_lc,
        llm,
        current_node,
        graph_state,
    )

    # Session completion summary
    logger.info("=== Session %d/%d complete: %d exchanges ===\n", session_num + 1, max_sessions, turn_count)

    return conversation_history_dict, updated_graph_state
