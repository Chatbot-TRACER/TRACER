"""Manages the conversational exploration sessions with the target chatbot."""

import enum
import secrets
from typing import TypedDict

from chatbot_connectors import Chatbot
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from tracer.analysis.functionality_extraction import extract_functionality_nodes
from tracer.analysis.functionality_refinement import (
    _process_node_group_for_merge,
    is_duplicate_functionality,
    merge_similar_functionalities,
    validate_parent_child_relationship,
)
from tracer.constants import (
    CONTEXT_MESSAGES_COUNT,
    MIN_CORRECTED_MESSAGE_LENGTH,
    MIN_EXPLORER_RESPONSE_LENGTH,
)
from tracer.prompts.session_prompts import (
    explorer_checker_prompt,
    get_correction_prompt,
    get_explorer_system_prompt,
    get_force_topic_change_instruction,
    get_initial_question_prompt,
    get_language_instruction,
    get_reminder_prompt,
    get_rephrase_prompt,
    get_session_focus,
    get_translation_prompt,
)
from tracer.schemas.functionality_node_model import FunctionalityNode
from tracer.utils.html_cleaner import clean_html_response
from tracer.utils.logging_utils import get_logger

from .conversation_utils import _get_all_nodes
from .fallback_detection import (
    is_semantically_fallback,
)

logger = get_logger()

MAX_LOG_MESSAGE_LENGTH = 500


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


def _remove_node_from_graph(node_name: str, graph_state: ExplorationGraphState) -> bool:
    """Removes a node and its children from the graph state.

    Args:
        node_name: The name of the node to remove
        graph_state: The current exploration graph state

    Returns:
        True if the node was found and removed, False otherwise
    """
    logger.debug("Attempting to remove node '%s' from graph", node_name)

    # Helper function to recursively find and remove a node
    def _find_and_remove_node(nodes: list[FunctionalityNode], target_name: str) -> tuple[bool, list[FunctionalityNode]]:
        remaining_nodes = []
        found = False

        for node in nodes:
            if node.name == target_name:
                found = True
                # If found, don't add to remaining_nodes
                logger.debug("Found node '%s' to remove", target_name)

                # Also remove from explored and pending nodes
                if target_name in graph_state["explored_nodes"]:
                    graph_state["explored_nodes"].remove(target_name)

                # Remove from pending nodes if present
                graph_state["pending_nodes"] = [n for n in graph_state["pending_nodes"] if n.name != target_name]

                # Don't process children - they'll be removed with the parent
                continue

            # Process this node's children
            if node.children:
                child_found, remaining_children = _find_and_remove_node(node.children, target_name)
                found = found or child_found

                # Update the node's children list with remaining children
                node.children = remaining_children

            # Keep this node
            remaining_nodes.append(node)

        return found, remaining_nodes

    # Start removal from root nodes
    found, graph_state["root_nodes"] = _find_and_remove_node(graph_state["root_nodes"], node_name)

    if found:
        logger.debug("Successfully removed node '%s' from graph", node_name)
    else:
        logger.warning("Failed to remove node '%s': Not found in graph", node_name)

    return found


def _analyze_session_and_update_nodes(
    conversation_history_lc: list[BaseMessage],
    llm: BaseLanguageModel,
    current_node: FunctionalityNode | None,
    graph_state: ExplorationGraphState,
) -> tuple[list[dict[str, str]], ExplorationGraphState]:
    """Analyzes conversation, extracts functionalities, and updates the exploration graph state."""
    conversation_history_dict = [
        {
            "role": "system" if isinstance(m, SystemMessage) else ("assistant" if isinstance(m, AIMessage) else "user"),
            "content": m.content,
        }
        for m in conversation_history_lc
    ]

    logger.debug("----------------------------------------------------------------------")
    logger.debug(
        "Starting Node Analysis for New Session. Current Node: %s",
        current_node.name if current_node else "None (Initial Exploration)",
    )
    logger.debug("----------------------------------------------------------------------")

    logger.verbose("\nAnalyzing conversation for new functionalities...")
    newly_extracted_nodes_raw = extract_functionality_nodes(conversation_history_dict, llm, current_node)
    session_added_or_merged_log_messages = []

    if newly_extracted_nodes_raw:
        # Create processing context
        context: NodeProcessingContext = {
            "current_node": current_node,
            "graph_state": graph_state,
            "llm": llm,
            "nodes_in_graph_before_this_session_processing": [],
            "already_merged_with_existing_names": set(),
        }

        session_added_or_merged_log_messages = _process_newly_extracted_nodes(newly_extracted_nodes_raw, context)

    if current_node:
        graph_state["explored_nodes"].add(current_node.name)

    _log_session_summary(session_added_or_merged_log_messages)

    return conversation_history_dict, graph_state


class NodeProcessingContext(TypedDict):
    """Context for processing nodes during session analysis."""

    current_node: FunctionalityNode | None
    graph_state: ExplorationGraphState
    llm: BaseLanguageModel
    nodes_in_graph_before_this_session_processing: list[FunctionalityNode]
    already_merged_with_existing_names: set[str]


def _process_newly_extracted_nodes(
    newly_extracted_nodes_raw: list[FunctionalityNode],
    context: NodeProcessingContext,
) -> list[str]:
    """Process newly extracted nodes by merging and adding them to the graph.

    Args:
        newly_extracted_nodes_raw: Raw extracted nodes from this session
        context: Processing context containing all necessary data

    Returns:
        List of log messages describing what was added or merged
    """
    session_added_or_merged_log_messages = []

    logger.debug(
        ">>> Raw Extracted Nodes from this Session (%d)",
        len(newly_extracted_nodes_raw),
    )
    for node in newly_extracted_nodes_raw:
        logger.debug(" • %s", node.name)

    # 1. Merge similar nodes discovered within this session's extractions
    logger.debug(">>> Performing Session-Local Merge on %d raw extracted nodes...", len(newly_extracted_nodes_raw))
    processed_new_nodes_this_session = merge_similar_functionalities(newly_extracted_nodes_raw, context["llm"])
    logger.debug(
        "<<< Nodes after Session-Local Merge (%d)",
        len(processed_new_nodes_this_session),
    )
    for node in processed_new_nodes_this_session:
        logger.debug(" • %s", node.name)

    # Initialize nodes list from graph state
    for root in context["graph_state"]["root_nodes"]:
        context["nodes_in_graph_before_this_session_processing"].extend(_get_all_nodes(root))

    for new_node_candidate in processed_new_nodes_this_session:
        log_messages = _process_single_node_candidate(new_node_candidate, context)
        session_added_or_merged_log_messages.extend(log_messages)

    # Global merge of root nodes after session additions.
    if context["graph_state"]["root_nodes"]:
        logger.debug(
            ">>> Performing Post-Session Merge on all %d Root Nodes...", len(context["graph_state"]["root_nodes"])
        )
        original_root_count = len(context["graph_state"]["root_nodes"])
        context["graph_state"]["root_nodes"] = merge_similar_functionalities(
            list(context["graph_state"]["root_nodes"]), context["llm"]
        )
        logger.debug(
            "<<< Root Nodes after Post-Session Merge (%d from %d).",
            len(context["graph_state"]["root_nodes"]),
            original_root_count,
        )

    return session_added_or_merged_log_messages


def _process_single_node_candidate(
    new_node_candidate: FunctionalityNode,
    context: NodeProcessingContext,
) -> list[str]:
    """Process a single node candidate for addition or merging.

    Args:
        new_node_candidate: The node candidate to process
        context: Processing context containing all necessary data

    Returns:
        List of log messages describing what was done
    """
    session_added_or_merged_log_messages = []

    logger.debug("--- Processing node for addition/merge: '%s' ---", new_node_candidate.name)

    # Check for duplicates against nodes ALREADY IN THE GRAPH from previous sessions
    is_dup, matching_existing_node_obj = is_duplicate_functionality(
        new_node_candidate,
        context["nodes_in_graph_before_this_session_processing"],  # Check only against established graph nodes
        context["llm"],
    )

    if is_dup and matching_existing_node_obj:
        log_messages = _handle_duplicate_node(new_node_candidate, matching_existing_node_obj, context)
        session_added_or_merged_log_messages.extend(log_messages)
    elif not is_dup:
        logger.debug(
            "Node '%s' is NOT a duplicate of any existing graph node. Attempting to add as new.",
            new_node_candidate.name,
        )
        was_added = _add_new_node_to_graph(
            new_node_candidate, context["current_node"], context["graph_state"], context["llm"]
        )
        if was_added:
            session_added_or_merged_log_messages.append(f"ADDED_NEW {new_node_candidate.name}")
            context["nodes_in_graph_before_this_session_processing"].append(new_node_candidate)
    else:
        logger.warning(
            "Node '%s' flagged as duplicate by LLM, but no specific matching node identified. Skipping for safety.",
            new_node_candidate.name,
        )

    return session_added_or_merged_log_messages


def _handle_duplicate_node(
    new_node_candidate: FunctionalityNode,
    matching_existing_node_obj: FunctionalityNode,
    context: NodeProcessingContext,
) -> list[str]:
    """Handle a duplicate node by merging or skipping.

    Args:
        new_node_candidate: The new node candidate
        matching_existing_node_obj: The existing node it matches
        context: Processing context containing all necessary data

    Returns:
        List of log messages describing what was done
    """
    session_added_or_merged_log_messages = []

    if matching_existing_node_obj.name in context["already_merged_with_existing_names"]:
        logger.debug(
            "Node '%s' is a duplicate of '%s', which has already been targeted for a merge this session. Skipping.",
            new_node_candidate.name,
            matching_existing_node_obj.name,
        )
        return session_added_or_merged_log_messages

    logger.debug(
        "Node '%s' is a duplicate of existing node '%s'. Attempting to merge them.",
        new_node_candidate.name,
        matching_existing_node_obj.name,
    )

    merge_candidates = [new_node_candidate, matching_existing_node_obj]
    merged_result_list = _process_node_group_for_merge(merge_candidates, context["llm"])

    if len(merged_result_list) == 1 and merged_result_list[0].name != matching_existing_node_obj.name:
        merged_node = merged_result_list[0]
        logger.debug(
            "Successfully merged '%s' and '%s' into '%s'. Updating graph.",
            new_node_candidate.name,
            matching_existing_node_obj.name,
            merged_node.name,
        )

        # Simplistic approach for now: Remove old, add new (potentially as root, hierarchy fixed later)
        _remove_node_from_graph(matching_existing_node_obj.name, context["graph_state"])
        _add_new_node_to_graph(merged_node, None, context["graph_state"], context["llm"])
        session_added_or_merged_log_messages.append(
            f"MERGED {new_node_candidate.name} with {matching_existing_node_obj.name} into {merged_node.name}"
        )
        context["already_merged_with_existing_names"].add(
            matching_existing_node_obj.name
        )  # Mark as processed for merge
        context["nodes_in_graph_before_this_session_processing"].append(
            merged_node
        )  # Add to list for subsequent checks
        if matching_existing_node_obj in context["nodes_in_graph_before_this_session_processing"]:
            context["nodes_in_graph_before_this_session_processing"].remove(matching_existing_node_obj)

    elif len(merged_result_list) == 1 and merged_result_list[0].name == matching_existing_node_obj.name:
        logger.debug(
            "Merge attempt between '%s' and '%s' resulted in keeping existing '%s' (no change or new info deemed better).",
            new_node_candidate.name,
            matching_existing_node_obj.name,
            matching_existing_node_obj.name,
        )
        session_added_or_merged_log_messages.append(
            f"CONSIDERED_MERGE {new_node_candidate.name} with {matching_existing_node_obj.name}, existing kept."
        )
        context["already_merged_with_existing_names"].add(matching_existing_node_obj.name)

    else:
        logger.debug(
            "Merge attempt between '%s' and '%s' decided to keep them separate. Adding '%s' as new.",
            new_node_candidate.name,
            matching_existing_node_obj.name,
            new_node_candidate.name,
        )
        was_added = _add_new_node_to_graph(
            new_node_candidate, context["current_node"], context["graph_state"], context["llm"]
        )
        if was_added:
            session_added_or_merged_log_messages.append(f"ADDED_NEW {new_node_candidate.name}")
            context["nodes_in_graph_before_this_session_processing"].append(new_node_candidate)

    return session_added_or_merged_log_messages


def _log_session_summary(session_added_or_merged_log_messages: list[str]) -> None:
    """Log the session summary.

    Args:
        session_added_or_merged_log_messages: List of log messages from the session
    """
    logger.info("--- Session Summary ---")
    if session_added_or_merged_log_messages:
        logger.info("Session processing resulted in %d additions/merges:", len(session_added_or_merged_log_messages))
        for msg in session_added_or_merged_log_messages:
            logger.info(" • %s", msg)
    else:
        logger.info("No new functionalities were added or merged in this session based on current graph state.")
    logger.debug("----------------------------------------------------------------------")
    logger.debug("Ending Node Analysis for Session.")
    logger.debug("----------------------------------------------------------------------\n")


def _get_explorer_next_message(
    conversation_history_lc: list[BaseMessage],
    llm: BaseLanguageModel,
    force_topic_change_instruction: str | None,
) -> str | None:
    """Gets the next message from the explorer LLM."""
    try:
        max_history_turns_for_llm = 10
        # Include system prompt + recent turns
        messages_for_llm = [conversation_history_lc[0], *conversation_history_lc[-(max_history_turns_for_llm * 2) :]]

        # Add periodic reminder about explorer role to prevent acting like the chatbot
        reminder_prompt = get_reminder_prompt()
        reminder_message = SystemMessage(content=reminder_prompt)

        # Add force topic change instruction if applicable
        if force_topic_change_instruction:
            messages_for_llm_this_turn = [
                *messages_for_llm,
                reminder_message,
                SystemMessage(content=force_topic_change_instruction),
            ]
        else:
            messages_for_llm_this_turn = [*messages_for_llm, reminder_message]

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

    # Clean HTML if present in the response
    cleaned_chatbot_message = clean_html_response(chatbot_message)

    original_chatbot_message = chatbot_message  # Store original for debugging
    chatbot_message = cleaned_chatbot_message  # Use the clean one

    is_initial_fallback = fallback_message and is_semantically_fallback(chatbot_message, fallback_message, llm)
    is_initial_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message
    needs_retry = is_initial_fallback or is_initial_parsing_error

    if not needs_retry:
        return InteractionOutcome.SUCCESS, chatbot_message

    # --- Retry Logic ---
    failure_reason = "Fallback message" if is_initial_fallback else "Potential chatbot error (OUTPUT_PARSING_FAILURE)"

    # Log the original message that caused the fallback
    logger.verbose(
        "Chatbot: %s",
        original_chatbot_message[:MAX_LOG_MESSAGE_LENGTH]
        + ("..." if len(original_chatbot_message) > MAX_LOG_MESSAGE_LENGTH else ""),
    )
    logger.verbose("(%s detected. Rephrasing and retrying...)", failure_reason)

    rephrase_prompt = get_rephrase_prompt(explorer_message)
    rephrased_message_or_original = explorer_message  # Default to original
    try:
        rephrased_response = llm.invoke(rephrase_prompt)
        rephrased_message = rephrased_response.content.strip().strip("\"'")
        if rephrased_message and rephrased_message != explorer_message:
            logger.debug("Rephrased original message")
            logger.verbose(
                "Explorer: %s",
                rephrased_message[:MAX_LOG_MESSAGE_LENGTH]
                + ("..." if len(rephrased_message) > MAX_LOG_MESSAGE_LENGTH else ""),
            )
            rephrased_message_or_original = rephrased_message
        else:
            logger.debug("Failed to generate a different rephrasing. Retrying with original.")
    except (ValueError, RuntimeError):
        logger.exception("Error rephrasing message. Retrying with original.")

    logger.debug("Sending retry message to chatbot")
    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(rephrased_message_or_original)

    if is_ok_retry:
        # Clean HTML in retry response if present
        cleaned_retry_message = clean_html_response(chatbot_message_retry)
        chatbot_message_retry = cleaned_retry_message

        is_retry_fallback = fallback_message and is_semantically_fallback(chatbot_message_retry, fallback_message, llm)
        is_retry_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message_retry
        if chatbot_message_retry != original_chatbot_message and not is_retry_fallback and not is_retry_parsing_error:
            logger.verbose("Retry successful with new response!")
            return InteractionOutcome.SUCCESS, chatbot_message_retry  # Success after retry

        logger.verbose("Retry failed (still received fallback/error or same message)")
        # Return original message but indicate retry failed
        return InteractionOutcome.RETRY_FAILED, chatbot_message

    logger.verbose("Retry failed (communication error)")
    # Return original message but indicate retry failed (due to comms)
    return InteractionOutcome.RETRY_FAILED, chatbot_message


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


def _check_explorer_impersonation(
    turn_count: int,
    explorer_response_content: str,
    conversation_history_lc: list[BaseMessage],
    llm: BaseLanguageModel,
) -> str:
    """Check if explorer is acting like a chatbot and correct if needed.

    Args:
        turn_count: Current turn number
        explorer_response_content: Explorer's response to check
        conversation_history_lc: Conversation history for context
        llm: Language model for checking and correction

    Returns:
        Corrected explorer response content
    """
    # Check if explorer is acting like a chatbot using LLM
    if turn_count > 1 and len(explorer_response_content) > MIN_EXPLORER_RESPONSE_LENGTH:
        try:
            check_prompt = [
                SystemMessage(content=explorer_checker_prompt()),
                HumanMessage(content=explorer_response_content),
            ]

            check_result = llm.invoke(check_prompt)
            is_chatbot_like = "YES" in check_result.content.upper() and "NO" not in check_result.content.upper()

            if is_chatbot_like:
                logger.warning("Explorer appears to be acting like a chatbot. Sending correction.")
                # Get last few messages for context
                latest_messages = (
                    conversation_history_lc[-CONTEXT_MESSAGES_COUNT:]
                    if len(conversation_history_lc) >= CONTEXT_MESSAGES_COUNT
                    else conversation_history_lc
                )

                correction_prompt = [
                    SystemMessage(content=get_correction_prompt()),
                    *latest_messages,
                    HumanMessage(
                        content=f"Original response: {explorer_response_content}\n\nRewrite this as a human user would say it:"
                    ),
                ]

                fixed_response = llm.invoke(correction_prompt)
                corrected_message = fixed_response.content.strip()

                if (
                    corrected_message
                    and len(corrected_message) > MIN_CORRECTED_MESSAGE_LENGTH
                    and corrected_message != explorer_response_content
                ):
                    logger.debug("Corrected explorer message from chatbot-like to user-like")
                    return corrected_message
        except Exception:
            logger.exception("Error in chatbot impersonation detection/correction")
            # Continue with original message if check fails

    return explorer_response_content


def _should_continue_conversation(
    turn_count: int,
    max_turns: int,
    explorer_response_content: str,
    final_chatbot_message: str,
    outcome: InteractionOutcome,
) -> bool:
    """Check if the conversation should continue.

    Args:
        turn_count: Current turn count
        max_turns: Maximum turns allowed
        explorer_response_content: Explorer's response
        final_chatbot_message: Chatbot's final message
        outcome: Interaction outcome

    Returns:
        True if conversation should continue, False otherwise
    """
    # Check max turns
    if turn_count >= max_turns:
        logger.debug("Reached maximum turns (%d). Ending session.", max_turns)
        return False

    # Check if explorer finished
    if "EXPLORATION COMPLETE" in explorer_response_content.upper():
        logger.debug("Explorer has finished session exploration")
        return False

    # Check communication error
    if outcome == InteractionOutcome.COMM_ERROR:
        return False

    # Check if chatbot ended conversation
    if final_chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
        logger.debug("Chatbot ended the conversation. Ending session.")
        return False

    return True


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

        # Check if explorer is acting like a chatbot and correct if needed
        explorer_response_content = _check_explorer_impersonation(
            turn_count, explorer_response_content, conversation_history_lc, llm
        )

        logger.verbose(
            "Explorer: %s",
            explorer_response_content[:MAX_LOG_MESSAGE_LENGTH]
            + ("..." if len(explorer_response_content) > MAX_LOG_MESSAGE_LENGTH else ""),
        )

        # Check if exploration is complete before adding to history
        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            logger.debug("Explorer has finished session exploration")
            conversation_history_lc.append(AIMessage(content=explorer_response_content))
            break

        # Add the explorer's message to history
        conversation_history_lc.append(AIMessage(content=explorer_response_content))

        # 4. Interact with Chatbot (with retry logic)
        outcome, final_chatbot_message = _handle_chatbot_interaction(
            explorer_response_content, the_chatbot, llm, fallback_message
        )

        # 5. Update History with chatbot's response
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

        # 7. Check if Conversation Should Continue
        should_continue = _should_continue_conversation(
            turn_count, max_turns, explorer_response_content, final_chatbot_message, outcome
        )
        if not should_continue:
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

    # Create a new conversation for this session
    new_conversation_success = the_chatbot.create_new_conversation()
    if new_conversation_success:
        logger.debug("Created new conversation for session %d", session_num + 1)
    else:
        logger.warning(
            "Failed to create new conversation for session %d, continuing with existing conversation", session_num + 1
        )

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
        # Clean HTML if present in the response
        original_message = chatbot_message  # Store for debugging if needed
        chatbot_message = clean_html_response(chatbot_message)

        # Log debug
        logger.debug("RAW message: %s", original_message)

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
    logger.info("\n=== Session %d/%d complete: %d exchanges ===\n", session_num + 1, max_sessions, turn_count)

    return conversation_history_dict, updated_graph_state
