import random
from typing import Any

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

from .conversation_utils import _get_all_nodes
from .fallback_detection import (
    is_semantically_fallback,
)


def _generate_initial_question(current_node, primary_language, llm):
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
        greeting_en = random.choice(possible_greetings)

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
            except Exception as e:
                print(f"Warning: Failed to translate initial greeting to {primary_language}: {e}")
                initial_question = greeting_en  # Fallback
        else:
            initial_question = greeting_en  # Use English

        print(f"   (Starting session 0 with general capability question: '{initial_question}')")
    return initial_question


def _analyze_session_and_update_nodes(
    conversation_history_lc: list[BaseMessage],
    llm: Any,
    current_node: FunctionalityNode | None,
    root_nodes: list[FunctionalityNode],
    pending_nodes: list[FunctionalityNode],
    explored_nodes: set[str],
) -> tuple[list[dict[str, str]], list[FunctionalityNode], list[FunctionalityNode], set[str]]:
    """Analyzes conversation, extracts/processes nodes, and updates node lists."""
    # Convert LangChain messages back to simple dicts
    conversation_history_dict = [
        {
            "role": "system" if isinstance(m, SystemMessage) else ("assistant" if isinstance(m, AIMessage) else "user"),
            "content": m.content,
        }
        for m in conversation_history_lc
    ]

    # Extract functionalities found in this session
    print("\nAnalyzing conversation for new functionalities...")
    new_functionality_nodes = extract_functionality_nodes(conversation_history_dict, llm, current_node)

    # Process newly found nodes
    if new_functionality_nodes:
        print(f"Discovered {len(new_functionality_nodes)} new functionality nodes:")
        new_functionality_nodes = merge_similar_functionalities(
            new_functionality_nodes, llm
        )  # Merge within session first
        all_existing_nodes = []
        for root in root_nodes:
            all_existing_nodes.extend(_get_all_nodes(root))

        for node in new_functionality_nodes:
            if not is_duplicate_functionality(node, all_existing_nodes, llm):
                is_added_as_child = False
                if current_node:
                    relationship_valid = validate_parent_child_relationship(current_node, node, llm)
                    if relationship_valid:
                        current_node.add_child(node)
                        print(f"  - '{node.name}' (child of '{current_node.name}')")
                        is_added_as_child = True

                if not is_added_as_child:
                    print(f"  - '{node.name}' (new root/standalone functionality)")
                    root_nodes.append(node)

                # Add to pending only if it's truly new (not duplicate) and not already explored
                if node.name not in explored_nodes:
                    pending_nodes.append(node)
                # Update all_existing_nodes for subsequent checks in this loop
                all_existing_nodes.append(node)  # Add the newly added node
            else:
                print(f"  - Skipped duplicate functionality: '{node.name}'")

        # Merge similar root nodes after adding new ones
        if root_nodes:
            root_nodes = merge_similar_functionalities(root_nodes, llm)

    # Mark the node we focused on as explored
    if current_node:
        explored_nodes.add(current_node.name)

    return conversation_history_dict, root_nodes, pending_nodes, explored_nodes


def _get_explorer_next_message(
    conversation_history_lc: list[BaseMessage],
    llm: Any,
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
        llm_response = llm.invoke(messages_for_llm_this_turn)
        return llm_response.content.strip()
    except Exception as e:
        print(f"\nError getting response from Explorer AI LLM: {e}. Ending session.")
        return None


def _handle_chatbot_interaction(
    explorer_message: str,
    the_chatbot: Any,
    llm: Any,
    fallback_message: str | None,
) -> tuple[bool, str, bool]:
    """Interacts with the chatbot, handles retries on fallback/error, and returns outcome.

    Returns:
        Tuple[bool, str, bool]: (is_comm_ok, final_chatbot_message, was_retry_needed_and_failed)
    """
    is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_message)

    if not is_ok:
        print("\nError communicating with chatbot.")
        return False, "[Chatbot communication error]", True  # Communication failed, counts as failed retry

    original_chatbot_message = chatbot_message
    is_initial_fallback = fallback_message and is_semantically_fallback(chatbot_message, fallback_message, llm)
    is_initial_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message
    retry_also_failed = False
    needs_retry = is_initial_fallback or is_initial_parsing_error

    if needs_retry:
        failure_reason = (
            "Fallback message" if is_initial_fallback else "Potential chatbot error (OUTPUT_PARSING_FAILURE)"
        )
        print(f"\n   ({failure_reason} detected. Rephrasing and retrying...)")
        rephrase_prompt = get_rephrase_prompt(explorer_message)
        try:
            rephrased_response = llm.invoke(rephrase_prompt)
            rephrased_message = rephrased_response.content.strip().strip("\"'")
            if rephrased_message and rephrased_message != explorer_message:
                print(f"   Original: '{explorer_message}'")
                print(f"   Rephrased: '{rephrased_message}'")
                is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(rephrased_message)
            else:
                print("   Failed to generate a different rephrasing. Retrying with original.")
                is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(explorer_message)
        except Exception as e:
            print(f"   Error rephrasing message: {e}. Retrying with original.")
            is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(explorer_message)

        if is_ok_retry:
            is_retry_fallback = fallback_message and is_semantically_fallback(
                chatbot_message_retry, fallback_message, llm
            )
            is_retry_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message_retry
            if (
                chatbot_message_retry != original_chatbot_message
                and not is_retry_fallback
                and not is_retry_parsing_error
            ):
                print("   Retry successful!")
                chatbot_message = chatbot_message_retry  # Use the successful retry message
                # retry_also_failed remains False
            else:
                print("   Retry failed (still received fallback/error or same message)")
                chatbot_message = original_chatbot_message  # Revert to original message on failed retry
                retry_also_failed = True
        else:
            print("   Retry failed (communication error)")
            chatbot_message = original_chatbot_message  # Revert to original message on failed retry
            retry_also_failed = True

    # Return communication status, the final message to record, and whether a needed retry failed
    return True, chatbot_message, retry_also_failed


def _run_conversation_loop(
    session_num: int,
    max_turns: int,
    llm: Any,
    the_chatbot: Any,
    fallback_message: str | None,
    initial_history: list[BaseMessage],
) -> tuple[list[BaseMessage], int]:
    """Runs the main conversation loop for a single session."""
    conversation_history_lc = initial_history[:]  # Work on a copy
    consecutive_failures = 0
    force_topic_change_next_turn = False
    turn_count = 1  # Start at 1 because the initial exchange is already in history

    while True:
        # 1. Check Stop Conditions
        if turn_count >= max_turns:
            print(f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}.")
            break

        # 2. Determine if Topic Change is Needed
        force_topic_change_instruction = get_force_topic_change_instruction(
            force_topic_change_next_turn, consecutive_failures
        )
        if force_topic_change_instruction:
            print(f"\n Forcing topic change: {force_topic_change_instruction.split(':')[1].strip()} !!!")
            # Reset flag after using it
            if force_topic_change_next_turn:
                force_topic_change_next_turn = False

        # 3. Get Explorer's Next Message
        explorer_response_content = _get_explorer_next_message(
            conversation_history_lc, llm, force_topic_change_instruction
        )
        if explorer_response_content is None:
            print("\nError getting response from Explorer AI LLM. Ending session.")
            break
        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            print(f"\nExplorer has finished session {session_num + 1} exploration.")
            break

        print(f"\nExplorer: {explorer_response_content}")
        conversation_history_lc.append(AIMessage(content=explorer_response_content))

        # 4. Interact with Chatbot (with retry logic)
        is_comm_ok, final_chatbot_message, retry_failed = _handle_chatbot_interaction(
            explorer_response_content, the_chatbot, llm, fallback_message
        )

        # 5. Update History and State based on Interaction Outcome
        conversation_history_lc.append(HumanMessage(content=final_chatbot_message))
        print(f"\nChatbot: {final_chatbot_message}")

        if not is_comm_ok:
            # Communication error during interaction (initial or retry)
            consecutive_failures += 1
            force_topic_change_next_turn = True  # Force change after comms error
            print(f"   (Communication failure. Consecutive failures: {consecutive_failures})")
            break  # End session on communication error

        # Check if the final response (after potential retry) is still a fallback/error
        final_is_fallback = fallback_message and is_semantically_fallback(final_chatbot_message, fallback_message, llm)
        final_is_parsing_error = "OUTPUT_PARSING_FAILURE" in final_chatbot_message

        if final_is_fallback or final_is_parsing_error:
            # Increment failures if the interaction ended in fallback/error
            consecutive_failures += 1
            print(f"   (Final response is fallback/error. Consecutive failures: {consecutive_failures})")
            # Force change only if a retry was attempted and it specifically failed
            if retry_failed:
                force_topic_change_next_turn = True
        else:
            # Successful interaction (or successful retry)
            if consecutive_failures > 0:
                print(f"   (Successful response. Resetting consecutive failures from {consecutive_failures}.)")
            consecutive_failures = 0
            force_topic_change_next_turn = False  # Ensure flag is reset on success

        # 6. Check if Chatbot Ended Conversation
        if final_chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
            print("Chatbot ended the conversation. Ending session.")
            break

        # 7. Increment Turn Count
        turn_count += 1

    return conversation_history_lc, turn_count


def run_exploration_session(
    session_num,
    max_sessions,
    max_turns,
    llm,
    the_chatbot,
    fallback_message: str | None = None,
    current_node: FunctionalityNode | None = None,
    explored_nodes: set[str] | None = None,
    pending_nodes: list[FunctionalityNode] | None = None,
    root_nodes: list[FunctionalityNode] | None = None,
    supported_languages=None,
):
    """Runs one chat session to explore the bot.

    Can focus on a specific 'current_node' if provided. Includes retry logic on fallback.

    Args:
        session_num (int): The current session number (0-based).
        max_sessions (int): Total sessions to run.
        max_turns (int): Max chat turns per session.
        llm: The language model instance.
        the_chatbot: The chatbot connector instance.
        fallback_message (str, optional): The detected fallback message of the chatbot. Defaults to None.
        current_node (FunctionalityNode, optional): Node to focus exploration on. Defaults to None.
        explored_nodes (set, optional): Set of names of already explored nodes. Defaults to None.
        pending_nodes (list, optional): List of nodes waiting to be explored. Defaults to None.
        root_nodes (list, optional): List of current root nodes. Defaults to None.
        supported_languages (list, optional): List of detected languages. Defaults to None.

    Returns:
        tuple: Contains the conversation history, detected languages (None),
               new nodes found this session,
               updated root nodes list, updated pending nodes list, updated explored nodes set.
    """
    # Setup default values if needed
    if explored_nodes is None:
        explored_nodes = set()
    if pending_nodes is None:
        pending_nodes = []
    if root_nodes is None:
        root_nodes = []
    if supported_languages is None:
        supported_languages = []

    print(f"\n--- Starting Exploration Session {session_num + 1}/{max_sessions} ---")
    if current_node:
        print(f"Exploring functionality: '{current_node.name}'")
    else:
        print("Exploring general capabilities")

    # Determine focus, language, system prompt
    session_focus = get_session_focus(current_node)
    primary_language = supported_languages[0] if supported_languages else "English"
    language_instruction = get_language_instruction(supported_languages, primary_language)
    system_content = get_explorer_system_prompt(session_focus, language_instruction, max_turns)

    # --- Initial Setup and First Exchange ---
    conversation_history_lc = [SystemMessage(content=system_content)]
    initial_question = _generate_initial_question(current_node, primary_language, llm)
    print(f"\nExplorer: {initial_question}")
    is_ok, chatbot_message = the_chatbot.execute_with_input(initial_question)
    print(f"\nChatbot: {chatbot_message}")

    turn_count = 0  # Initialize turn count
    if not is_ok:
        print("\nError communicating with chatbot on initial message. Ending session.")
        conversation_history_lc.append(AIMessage(content=initial_question))
        conversation_history_lc.append(HumanMessage(content="[Chatbot communication error on initial message]"))
        # turn_count remains 0
    else:
        conversation_history_lc.append(AIMessage(content=initial_question))
        conversation_history_lc.append(HumanMessage(content=chatbot_message))
        turn_count = 1  # First exchange successful

        # --- Run the main conversation loop ---
        if turn_count < max_turns:  # Only run loop if max_turns allows more turns
            conversation_history_lc, turn_count = _run_conversation_loop(
                session_num=session_num,
                max_turns=max_turns,
                llm=llm,
                the_chatbot=the_chatbot,
                fallback_message=fallback_message,
                initial_history=conversation_history_lc,
            )

    # --- Post-Session Analysis ---
    (
        conversation_history_dict,
        root_nodes,
        pending_nodes,
        explored_nodes,
    ) = _analyze_session_and_update_nodes(
        conversation_history_lc,
        llm,
        current_node,
        root_nodes,
        pending_nodes,
        explored_nodes,
    )

    print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")

    # Return all updated state
    return (
        conversation_history_dict,
        root_nodes,
        pending_nodes,
        explored_nodes,
    )
