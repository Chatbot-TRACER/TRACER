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
from chatbot_explorer.utils.html_cleaner import clean_html_response
from chatbot_explorer.utils.logging_utils import get_logger
from connectors.chatbot_connectors import Chatbot

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
    session_added_nodes = []

    if newly_extracted_nodes_raw:
        logger.debug(">>> Raw Extracted Nodes from this Session (%d):", len(newly_extracted_nodes_raw))
        for i, node in enumerate(newly_extracted_nodes_raw):
            logger.debug("  RAW_EXTRACTED[%d]: Name: %s, Desc: %s...", i, node.name, node.description[:50])

        # 1. Merge similar nodes discovered *within this session's extractions*
        logger.debug(">>> Performing Session-Local Merge on %d raw extracted nodes...", len(newly_extracted_nodes_raw))
        processed_new_nodes_this_session = merge_similar_functionalities(newly_extracted_nodes_raw, llm)
        logger.debug("<<< Nodes after Session-Local Merge (%d):", len(processed_new_nodes_this_session))
        for i, node in enumerate(processed_new_nodes_this_session):
            logger.debug("  SESSION_MERGED[%d]: Name: %s, Desc: %s...", i, node.name, node.description[:50])

        # Get all unique nodes currently in the graph for duplicate checking.
        all_existing_nodes_in_graph = []
        for root in graph_state["root_nodes"]:
            all_existing_nodes_in_graph.extend(_get_all_nodes(root))  # Assuming _get_all_nodes flattens the tree
        logger.debug("Found %d existing nodes in graph for duplicate checking.", len(all_existing_nodes_in_graph))

        nodes_added_in_current_pass = []  # For duplicate checking within this session's batch

        for new_node in processed_new_nodes_this_session:
            logger.debug("--- Processing node for addition: '%s' ---", new_node.name)
            # 2. Check if this new_node is a duplicate of anything already in the graph or added this pass.
            combined_check_list = all_existing_nodes_in_graph + nodes_added_in_current_pass
            logger.debug(
                "Checking for duplicates of '%s' against %d existing/session-added nodes.",
                new_node.name,
                len(combined_check_list),
            )
            is_dup = is_duplicate_functionality(
                new_node, combined_check_list, llm
            )  # is_duplicate_functionality should log its reasoning

            if not is_dup:
                logger.debug("Node '%s' is NOT a duplicate. Attempting to add to graph.", new_node.name)
                was_added = _add_new_node_to_graph(
                    new_node, current_node, graph_state, llm
                )  # _add_new_node_to_graph handles parent/child linking
                if was_added:
                    logger.debug("Successfully added '%s' to the graph.", new_node.name)
                    nodes_added_in_current_pass.append(new_node)
                    session_added_nodes.append(new_node)
                    # Update all_existing_nodes_in_graph to include the newly added node for subsequent checks in this loop
                    all_existing_nodes_in_graph.append(new_node)
                else:
                    logger.warning(
                        "Failed to add node '%s' to graph (e.g., relationship validation failed).", new_node.name
                    )
            else:
                logger.debug("Skipped adding node '%s' as it was identified as a duplicate.", new_node.name)

        # Optional: Global merge of root nodes after session additions.
        # This can be heavy. Consider if this is better done less frequently or as part of the end-of-exploration phase.
        if session_added_nodes and graph_state["root_nodes"]:  # Only if new nodes were added AND there are roots
            logger.debug(">>> Performing Post-Session Merge on all %d Root Nodes...", len(graph_state["root_nodes"]))
            original_root_names = [r.name for r in graph_state["root_nodes"]]
            graph_state["root_nodes"] = merge_similar_functionalities(
                list(graph_state["root_nodes"]), llm
            )  # Pass a copy
            logger.debug(
                "<<< Root Nodes after Post-Session Merge (%d). Original roots: %s",
                len(graph_state["root_nodes"]),
                original_root_names,
            )
            current_root_names = [r.name for r in graph_state["root_nodes"]]
            if original_root_names != current_root_names:  # Check if any change happened
                logger.debug("  Current roots: %s", current_root_names)

    if current_node:
        graph_state["explored_nodes"].add(current_node.name)

    logger.info("--- Session Summary ---")
    if session_added_nodes:
        logger.info("\nDiscovered and added %d new unique functionalities in this session:", len(session_added_nodes))
        for node in session_added_nodes:
            # Finding parentage for logging can be tricky here as graph_state is mutable
            # This is a simplified way to show context
            relation_to_current = (
                " (related to current node exploration)" if current_node and node.parent == current_node else ""
            )
            logger.info(" • ADDED: %s %s", node.name, relation_to_current)
    else:
        logger.info("No new unique functionalities were added to the graph in this session.")
    logger.debug("----------------------------------------------------------------------")
    logger.debug(
        "Ending Node Analysis for Session. Current Node: %s",
        current_node.name if current_node else "None (Initial Exploration)",
    )
    logger.debug("----------------------------------------------------------------------\n")

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
        logger.verbose(
            "Explorer: %s",
            explorer_response_content[:MAX_LOG_MESSAGE_LENGTH]
            + ("..." if len(explorer_response_content) > MAX_LOG_MESSAGE_LENGTH else ""),
        )

        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            logger.debug("Explorer has finished session exploration")
            conversation_history_lc.append(AIMessage(content=explorer_response_content))
            break

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
        # Clean HTML if present in the response
        from chatbot_explorer.utils.html_cleaner import clean_html_response

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
