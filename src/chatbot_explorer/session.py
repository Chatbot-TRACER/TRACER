import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .analysis.functionality_extraction import extract_functionality_nodes
from .analysis.node_processing import (
    is_duplicate_functionality,
    merge_similar_functionalities,
    validate_parent_child_relationship,
)
from .conversation.conversation_utils import _get_all_nodes
from .conversation.fallback_detection import (
    is_semantically_fallback,
)
from .schemas.functionality_node import FunctionalityNode


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

    # Determine the focus for this session
    if current_node:
        # Focus on the specific node
        session_focus = f"Focus on actively using and exploring the '{current_node.name}' functionality ({current_node.description}). If it requires input, try providing plausible values. If it offers choices, select one to proceed."
        if current_node.parameters:
            param_names = [p.get("name", "unknown") for p in current_node.parameters]
            session_focus += f" Attempt to provide values for parameters like: {', '.join(param_names)}."
    else:
        # General exploration focus
        session_focus = "Explore the chatbot's main capabilities. Ask what it can do or what topics it covers. If it offers options or asks questions requiring a choice, TRY to provide an answer or make a selection to see where it leads."

    # Determine primary language for interaction
    primary_language = supported_languages[0] if supported_languages else "English"
    lang_lower = primary_language.lower()

    # Add language info to system prompt
    language_instruction = ""
    if supported_languages:
        language_str = ", ".join(supported_languages)
        language_instruction = f"\n\nIMPORTANT: The chatbot supports these languages: {language_str}. YOU MUST COMMUNICATE PRIMARILY IN {language_str}."

    # System prompt for the explorer AI
    system_content = f"""You are an Explorer AI tasked with actively discovering and testing the capabilities of another chatbot through conversation. Your goal is to map out its functionalities and interaction flows.

    IMPORTANT GUIDELINES:
    1. Ask ONE clear question or give ONE clear instruction/command at a time.
    2. Keep messages concise but focused on progressing the interaction or using a feature according to the current focus.
    3. **CRITICAL: If the chatbot offers clear interactive choices (e.g., buttons, numbered lists, "Option A or Option B?", "Yes or No?"), you MUST try to select one of the offered options in your next turn to explore that path.**
    4. **ADAPTIVE EXPLORATION (Handling Non-Progressing Turns):**
        - **If the chatbot provides information (like an explanation, contact details, status update) OR a fallback/error message, and does NOT ask a question or offer clear interactive choices:**
            a) **Check for Repetitive Failure on the SAME GOAL:** If the chatbot has given the **same or very similar fallback/error message** for the last **2** turns despite you asking relevant questions about the *same underlying topic or goal*, **DO NOT REPHRASE the failed question/request again**. Instead, **ABANDON this topic/goal for this session**. Your next turn MUST be to ask about a **completely different capability** or topic you know exists or is plausible (e.g., switch from asking about custom pizza ingredients to asking about predefined pizzas or drinks), OR if no other path is obvious, respond with "EXPLORATION COMPLETE".
            b) **If NOT Repetitive Failure (e.g., first fallback on this topic):** Ask a specific, relevant clarifying question about the information/fallback provided ONLY IF it seems likely to yield progress. Otherwise, or if clarification isn't obvious, **switch to a NEW, specific, plausible topic/task** relevant to the chatbot's likely domain (infer this domain). **Avoid simply rephrasing the previous failed request.** Do NOT just ask "What else?".
        - **Otherwise (if the bot asks a question or offers choices):** Respond appropriately to continue the current flow or make a selection as per Guideline 3.
    5. Prioritize actions/questions relevant to the `EXPLORATION FOCUS` below.
    6. Follow the chatbot's conversation flow naturally. {language_instruction}

    EXPLORATION FOCUS FOR THIS SESSION:
    {session_focus}

    Try to follow the focus and the adaptive exploration guideline, especially the rule about abandoning topics after repetitive failures. After {max_turns} exchanges, or when you believe you have thoroughly explored this specific path/topic (or reached a dead end/loop), respond ONLY with "EXPLORATION COMPLETE".
    """

    # Start fresh conversation history
    conversation_history_lc = [SystemMessage(content=system_content)]

    # Generate the first question
    if current_node:
        # Ask about the specific node
        question_prompt = f"""
        You need to generate an initial question/command to start exploring a specific chatbot functionality.

        FUNCTIONALITY TO EXPLORE:
        Name: {current_node.name}
        Description: {current_node.description}
        Parameters: {", ".join(p.get("name", "?") for p in current_node.parameters) if current_node.parameters else "None"}

        {"IMPORTANT: Generate your question/command in " + primary_language + "." if primary_language else ""}

        Generate a simple, direct question or command relevant to initiating this functionality.
        Example: If exploring 'provide_contact_info', ask 'How can I contact support?' or 'What is the support email?'.
        """
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
                translation_prompt = (
                    f"Translate '{greeting_en}' to {primary_language}. Respond ONLY with the translation."
                )
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

    print(f"\nExplorer: {initial_question}")

    # Send first question to the chatbot
    is_ok, chatbot_message = the_chatbot.execute_with_input(initial_question)
    print(f"\nChatbot: {chatbot_message}")

    # Add first exchange to history
    conversation_history_lc.append(AIMessage(content=initial_question))  # Explorer AI is 'assistant'
    conversation_history_lc.append(HumanMessage(content=chatbot_message))  # Target Chatbot is 'user'/'human'

    consecutive_failures = 0
    force_topic_change_next_turn = False

    # Main loop
    turn_count = 1  # We already did the first question, so start at 1
    while True:
        # Stop if we hit the max number of turns
        if turn_count >= max_turns:
            print(f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}.")
            break

        # --- Check for forcing topic change (due to consecutive failures OR failed retry) ---
        force_topic_change_instruction = None
        # Check flag from previous turn's failed retry first
        if force_topic_change_next_turn:
            force_topic_change_instruction = "CRITICAL OVERRIDE: Your previous attempt AND a retry both failed (likely hit fallback). You MUST abandon the last topic/question now. Ask about a completely different, plausible capability"
            print("\n Forcing topic change: Retry failed previously. !!!")
            force_topic_change_next_turn = False  # Reset flag after using it
        # Then check consecutive failures (if retry didn't trigger it)
        elif consecutive_failures >= 2:
            force_topic_change_instruction = f"CRITICAL OVERRIDE: The chatbot has failed to respond meaningfully {consecutive_failures} times in a row on the current topic/line of questioning. You MUST abandon this topic now. Ask about a completely different, plausible capability"
            print(f"\n Forcing topic change: {consecutive_failures} consecutive failures. !!!")
        # ---

        # --- Get what the explorer wants to say next ---
        explorer_response_content = None
        try:
            max_history_turns_for_llm = 10  # Keep last 10 turns (20 messages) + system prompt
            messages_for_llm = [conversation_history_lc[0]] + conversation_history_lc[
                -(max_history_turns_for_llm * 2) :
            ]

            # If forcing change, add a temporary system message for this turn
            if force_topic_change_instruction:
                messages_for_llm_this_turn = [*messages_for_llm, SystemMessage(content=force_topic_change_instruction)]
            else:
                messages_for_llm_this_turn = messages_for_llm

            # Invoke the LLM directly
            llm_response = llm.invoke(messages_for_llm_this_turn)
            explorer_response_content = llm_response.content.strip()

        except Exception as e:
            print(f"\nError getting response from Explorer AI LLM: {e}. Ending session.")
            break  # Stop if LLM fails

        if not explorer_response_content:
            print("\nError: Failed to get next action from Explorer AI LLM. Ending session.")
            break

        print(f"\nExplorer: {explorer_response_content}")

        # If the explorer says it's done, just stop
        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            print(f"\nExplorer has finished session {session_num + 1} exploration.")
            break

        # Save what the explorer said before sending it to the chatbot
        conversation_history_lc.append(AIMessage(content=explorer_response_content))

        # --- Send the explorer's message to the chatbot ---
        is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_response_content)

        if not is_ok:
            print("\nError communicating with chatbot. Ending session.")
            conversation_history_lc.append(HumanMessage(content="[Chatbot communication error]"))
            consecutive_failures += 1
            force_topic_change_next_turn = True
            break

        # Save the chatbot's first response in case we need it later
        original_chatbot_message = chatbot_message

        # Check if the chatbot gave us a fallback or error
        is_fallback = False
        if fallback_message and chatbot_message:
            # Use LLM for semantic comparison
            is_fallback = is_semantically_fallback(chatbot_message, fallback_message, llm)

        is_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message

        # --- Try rephrasing the message if we hit a fallback or parsing error ---
        retry_also_failed = False
        if is_fallback or is_parsing_error:
            failure_reason = "Fallback message" if is_fallback else "Potential chatbot error (OUTPUT_PARSING_FAILURE)"
            print(f"\n   ({failure_reason} detected. Rephrasing and retrying...)")

            # Generate a rephrased version of the original message
            rephrase_prompt = f"""
            The chatbot did not understand this message: "{explorer_response_content}"

            Please rephrase this message to convey the same intent but with different wording.
            Make the rephrased version simpler, more direct, and avoid complex structures.
            ONLY return the rephrased message, nothing else.
            """

            try:
                rephrased_response = llm.invoke(rephrase_prompt)
                rephrased_message = rephrased_response.content.strip().strip("\"'")

                if rephrased_message and rephrased_message != explorer_response_content:
                    print(f"   Original: '{explorer_response_content}'")
                    print(f"   Rephrased: '{rephrased_message}'")

                    # Try with the rephrased message
                    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(rephrased_message)
                else:
                    # Fallback to original if rephrasing failed or returned identical text
                    print("   Failed to generate a different rephrasing. Retrying with original.")
                    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(explorer_response_content)
            except Exception as e:
                print(f"   Error rephrasing message: {e}. Retrying with original.")
                is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(explorer_response_content)

            if is_ok_retry:
                # See if the retry gave us something different and not another failure
                is_retry_fallback = False
                if fallback_message and chatbot_message_retry:
                    is_retry_fallback = is_semantically_fallback(chatbot_message_retry, fallback_message, llm)

                is_retry_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message_retry

                if (
                    chatbot_message_retry != original_chatbot_message
                    and not is_retry_fallback
                    and not is_retry_parsing_error
                ):
                    # Retry worked, use the new response
                    print("   Retry successful!")
                    chatbot_message = chatbot_message_retry
                    consecutive_failures = 0
                else:
                    # Retry didn't help, just use the original
                    print("   Retry failed (still received fallback/error)")
                    chatbot_message = original_chatbot_message
                    retry_also_failed = True
            else:
                # Retry couldn't even talk to the chatbot
                print("   Retry failed (communication error)")
                chatbot_message = original_chatbot_message
                retry_also_failed = True
        # --- END Rephrasing Retry Logic ---

        # --- Update state based on FINAL outcome of the turn ---
        final_is_fallback = False
        if fallback_message and chatbot_message:
            final_is_fallback = is_semantically_fallback(chatbot_message, fallback_message, llm)
        final_is_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message

        if final_is_fallback or final_is_parsing_error:
            # Increment consecutive failures ONLY IF the retry didn't already succeed
            if not (is_fallback or is_parsing_error) or retry_also_failed:
                consecutive_failures += 1
                print(f"   (Consecutive failures: {consecutive_failures})")
            # Set flag to force change next turn IF the retry specifically failed
            if retry_also_failed:
                force_topic_change_next_turn = True
        else:
            # Reset counter if the turn was successful
            if consecutive_failures > 0:
                print(
                    f"   (Successful response this turn. Resetting consecutive failures from {consecutive_failures}.)",
                )
            consecutive_failures = 0
            force_topic_change_next_turn = False  # Ensure flag is off on success
        # ---

        print(f"\nChatbot: {chatbot_message}")

        conversation_history_lc.append(HumanMessage(content=chatbot_message))

        if chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
            print("Chatbot ended the conversation. Ending session.")
            break

        turn_count += 1

    # Convert LangChain messages back to simple dicts for analysis functions if needed
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

        # Merge similar nodes found *within this session* first
        new_functionality_nodes = merge_similar_functionalities(new_functionality_nodes, llm)

        for node in new_functionality_nodes:
            # Check against *all* nodes found so far
            all_existing = []
            for root in root_nodes:
                all_existing.extend(_get_all_nodes(root))  # Get all descendants

            if not is_duplicate_functionality(node, all_existing, llm):
                # If exploring a specific node, check if the new one is related
                if current_node:
                    relationship_valid = validate_parent_child_relationship(current_node, node, llm)

                    if relationship_valid:
                        # Add as child if valid relationship
                        current_node.add_child(node)
                        print(f"  - '{node.name}' (child of '{current_node.name}')")
                    else:
                        # Add as a new root if not related
                        print(f"  - '{node.name}' (standalone functionality)")
                        root_nodes.append(node)
                else:
                    # Add as a new root if not exploring a specific node
                    root_nodes.append(node)
                    print(f"  - '{node.name}' (root node for now)")

                if node.name not in explored_nodes:
                    pending_nodes.append(node)
            else:
                print(f"  - Skipped duplicate functionality: '{node.name}'")

        # Merge similar root nodes after adding new ones
        if root_nodes:
            root_nodes = merge_similar_functionalities(root_nodes, llm)

    # Mark the node we focused on as explored
    if current_node:
        explored_nodes.add(current_node.name)

    print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")

    # Return all updated state
    return (
        conversation_history_dict,
        root_nodes,  # Updated list of roots
        pending_nodes,  # Updated pending queue
        explored_nodes,  # Updated set of explored names
    )
