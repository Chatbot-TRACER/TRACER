from typing import List, Dict, Any, Optional, Tuple, Set
from .functionality_node import FunctionalityNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re
import json
import random


def extract_supported_languages(chatbot_response, llm):
    """
    Try to figure out what languages the chatbot knows.

    Args:
        chatbot_response (str): The chatbot's message.
        llm: The language model instance.

    Returns:
        list: A list of language names (strings).
    """
    language_prompt = f"""
    Based on the following chatbot response, determine what language(s) the chatbot supports.
    If the response is in a non-English language, include that language in the list.
    If the response explicitly mentions supported languages, list those.

    CHATBOT RESPONSE:
    {chatbot_response}

    FORMAT YOUR RESPONSE AS A COMMA-SEPARATED LIST OF LANGUAGES:
    [language1, language2, ...]

    RESPONSE:
    """

    language_result = llm.invoke(language_prompt)
    languages = language_result.content.strip()

    # Clean up the LLM response
    languages = languages.replace("[", "").replace("]", "")
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    return language_list


def extract_fallback_message(the_chatbot, llm):
    """
    Try to get the chatbot's "I don't understand" message.
    Sends confusing messages to trigger it. These aren't part of the main chat history.

    Args:
        the_chatbot: The chatbot connector instance.
        llm: The language model instance.

    Returns:
        str or None: The detected fallback message, or None if not found.
    """
    print(
        "\n--- Attempting to detect chatbot fallback message (won't be included in analysis) ---"
    )

    # Some weird questions to confuse the bot
    confusing_queries = [
        "What is the square root of a banana divided by the color blue?",
        "Please explain quantum chromodynamics in terms of medieval poetry",
        "Xyzzplkj asdfghjkl qwertyuiop?",
        "If tomorrow's yesterday was three days from now, how many pancakes fit in a doghouse?",
        "Can you please recite the entire source code of Linux kernel version 5.10?",
    ]

    responses = []

    # Send confusing queries and get responses
    for i, query in enumerate(confusing_queries):
        print(f"\nSending confusing query {i + 1}...")
        try:
            is_ok, response = the_chatbot.execute_with_input(query)

            if is_ok:
                print(f"Response received ({len(response)} chars)")
                responses.append(response)
        except Exception as e:
            print(f"Error communicating with chatbot: {e}")

    # Analyze responses if we got any
    if responses:
        analysis_prompt = f"""
        I'm trying to identify a chatbot's fallback message - the standard response it gives when it doesn't understand.

        Below are responses to intentionally confusing or nonsensical questions.
        If there's a consistent pattern or identical response, that's likely the fallback message.

        RESPONSES:
        {responses}

        ANALYSIS STEPS:
        1. Check for identical responses - if any responses are exactly the same, that's likely the fallback.
        2. Look for very similar responses with only minor variations.
        3. Identify common phrases or sentence patterns across responses.

        EXTRACT ONLY THE MOST LIKELY FALLBACK MESSAGE OR PATTERN.
        If the fallback message appears to have minor variations, extract the common core part that appears in most responses.
        Do not include any analysis, explanation, or quotation marks in your response.
        """

        try:
            fallback_result = llm.invoke(analysis_prompt)
            fallback = fallback_result.content

            # Clean up the fallback message
            fallback = fallback.strip()
            # Remove quotes at beginning and end if present
            fallback = re.sub(r'^["\']+|["\']+$', "", fallback)
            # Remove any "Fallback message:" prefix if the LLM included it
            fallback = re.sub(
                r"^(Fallback message:?\s*)", "", fallback, flags=re.IGNORECASE
            )

            if fallback:
                print(
                    f'Detected fallback pattern: "{fallback[:50]}{"..." if len(fallback) > 50 else ""}"'
                )
                return fallback
            else:
                print("Could not extract a clear fallback message pattern.")
        except Exception as e:
            print(f"Error during fallback analysis: {e}")

    print("Could not detect a consistent fallback message.")
    return None


def is_semantically_fallback(response: str, fallback: str, llm) -> bool:
    """
    Uses LLM to determine if a chatbot response is semantically equivalent
    to a known fallback message.

    Args:
        response (str): The chatbot's current response.
        fallback (str): The known fallback message pattern.
        llm: The language model instance.

    Returns:
        bool: True if the response is considered a fallback, False otherwise.
    """
    if not response or not fallback:
        return False  # Cannot compare if one is empty

    prompt = f"""
    Compare the following two messages. Determine if the "Chatbot Response" is semantically equivalent to the "Known Fallback Message".

    "Semantically equivalent" means the response conveys the same core meaning as the fallback, such as:
    - Not understanding the request.
    - Being unable to process the request.
    - Asking the user to rephrase.
    - Stating a general limitation.

    It does NOT have to be an exact word-for-word match.

    Known Fallback Message:
    "{fallback}"

    Chatbot Response:
    "{response}"

    Is the "Chatbot Response" semantically equivalent to the "Known Fallback Message"?

    Respond with ONLY "YES" or "NO".
    """
    try:
        llm_decision = llm.invoke(prompt)
        decision_text = llm_decision.content.strip().upper()

        return decision_text.startswith("YES")
    except Exception as e:
        print(f"   LLM Fallback Check Error: {e}. Assuming not a fallback.")
        return False  # Default to False if LLM fails


def extract_functionality_nodes(conversation_history, llm, current_node=None):
    """
    Pull out FunctionalityNodes from the conversation.
    Tries to find steps in the user's workflow.

    Args:
        conversation_history (list): The list of chat messages.
        llm: The language model instance.
        current_node (FunctionalityNode, optional): The node being explored. Defaults to None.

    Returns:
        list: A list of newly found FunctionalityNode objects.
    """
    # Format conversation for the LLM
    formatted_conversation = format_conversation(conversation_history)

    # Context for the LLM
    context = "Identify distinct interaction steps or functionalities the chatbot provides in this conversation, relevant to the user's workflow."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' step: {current_node.description}"

    # Prompt for the LLM
    extraction_prompt = f"""
    {context}

    CONVERSATION:
    {formatted_conversation}

    Analyze the conversation and extract distinct **chatbot capabilities, actions performed, or information provided by the chatbot**.

    **CRITICAL RULES:**
    1.  **Focus ONLY on the CHATBOT:** Functionalities MUST represent actions performed *by the chatbot* or information *provided by the chatbot*.
    2.  **EXCLUDE User Actions:** DO NOT extract steps that only describe what the *user* asks, says, or does (e.g., 'user_asks_for_help', 'user_selects_option', 'user_provides_details').
    3.  **Chatbot's Perspective:** If a user action triggers a chatbot response or workflow, name and describe the functionality based on the **chatbot's role** in handling that trigger (e.g., `handle_order_request`, `collect_user_selection`, `prompt_for_details`, `provide_requested_info`, `confirm_details`).
    4.  **Naming:** Use clear, descriptive snake_case names reflecting the chatbot's action (e.g., `provide_menu`, `collect_size`, `confirm_order`, `process_payment`).
    5.  **Avoid Failures:** AVOID extracting functionalities that solely describe the chatbot failing to understand or provide information (e.g., 'handle_repeat_requests', 'fail_to_provide_info'). Focus on successful actions or information provided, or capabilities the chatbot *claims* to have.

    **EXAMPLES (Focus on Chatbot Actions):**

    - **Scenario: Booking Appointment (Transactional)**
        - User: "I need to book an appointment."
        - Chatbot: "Okay, what date are you looking for?"
        - User: "Next Tuesday."
        - Chatbot: "We have slots at 10 AM and 2 PM available."
        - User: "10 AM works."
        - Chatbot: "Great, I need your email to confirm."
        - User: "test@example.com"
        - Chatbot: "Confirmed for Tuesday at 10 AM. Email: test@example.com."

        - **BAD Extractions (User Actions):**
            - `user_requests_booking` (User action)
            - `user_provides_date` (User action)
            - `user_selects_time` (User action)
            - `user_gives_email` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `prompt_for_booking_date` (Chatbot asks for date)
            - `provide_available_time_slots` (Chatbot lists options)
            - `prompt_for_confirmation_email` (Chatbot asks for email)
            - `confirm_booking_details` (Chatbot confirms the final booking)

    - **Scenario: Company Policy Info (Informational)**
        - User: "What's the policy on remote work?"
        - Chatbot: "Our remote work policy allows eligible employees to work remotely up to 3 days per week, subject to manager approval. More details are in the employee handbook."
        - User: "Where can I find the handbook?"
        - Chatbot: "You can find the employee handbook on the company intranet under HR Documents."

        - **BAD Extractions (User Actions):**
            - `user_asks_remote_policy` (User action)
            - `user_asks_for_handbook_location` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `explain_remote_work_policy` (Chatbot provides policy summary)
            - `provide_handbook_location` (Chatbot tells where to find the handbook)

    - **Scenario: General Inquiry (Informational/Meta)**
        - User: "What can you do?"
        - Chatbot: "I can help you book appointments, check order status, and answer questions about our services."

        - **BAD Extractions (User Actions):**
            - `user_asks_capabilities` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `list_main_capabilities` (Chatbot lists what it can do)


    For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
    1. A short, descriptive name (snake_case, from chatbot's perspective).
    2. A clear description (from chatbot's perspective - what the chatbot DOES or PROVIDES).
    3. Required parameters (inputs the chatbot NEEDS for this step, comma-separated or "None").

    Format EXACTLY as:
    FUNCTIONALITY:
    name: chatbot_action_name
    description: What the chatbot does or provides.
    parameters: param1, param2 (or "None")

    If no new relevant **chatbot capability/action** is identified, respond ONLY with "NO_NEW_FUNCTIONALITY".
    """
    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    print("\n--- Raw LLM Response for Functionality Extraction ---")
    print(content)
    print("-----------------------------------------------------")

    # Parse the LLM response
    functionality_nodes = []

    if "NO_NEW_FUNCTIONALITY" in content.upper():  # Case-insensitive check
        print("  LLM indicated no new functionalities.")
        return functionality_nodes

    # Split response into blocks
    blocks = re.split(r"FUNCTIONALITY:\s*", content, flags=re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block:  # Skip empty parts
            continue

        name = None
        description = None
        params_str = "None"

        # Parse lines in the block
        lines = block.split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("name:"):
                name = line[len("name:") :].strip()
            elif line.lower().startswith("description:"):
                description = line[len("description:") :].strip()
            elif line.lower().startswith("parameters:"):
                params_str = line[len("parameters:") :].strip()

        # Create node if we got name and description
        if name and description:
            # Parse parameters string
            parameters = []
            if params_str.lower() != "none":
                param_names = [p.strip() for p in params_str.split(",") if p.strip()]
                # Basic parameter structure
                parameters = [
                    {"name": p, "type": "string", "description": f"Parameter {p}"}
                    for p in param_names
                ]

            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                parent=current_node,  # Set parent for now
            )
            functionality_nodes.append(new_node)
            print(f"  Identified step (Robust Parsing): {name}")
        elif block:  # Log blocks we couldn't parse
            print(f"  WARN: Could not parse functionality block:\n{block}")

    if not functionality_nodes and "NO_NEW_FUNCTIONALITY" not in content.upper():
        print(
            "  WARN: LLM response did not contain 'NO_NEW_FUNCTIONALITY' but no functionalities were parsed."
        )

    return functionality_nodes


def is_duplicate_functionality(
    node: FunctionalityNode, existing_nodes: List[FunctionalityNode], llm=None
) -> bool:
    """
    Check if this node is basically the same as one we already have.

    Args:
        node (FunctionalityNode): The new node to check.
        existing_nodes (list): List of nodes already found.
        llm (optional): The language model instance for semantic check. Defaults to None.

    Returns:
        bool: True if it seems like a duplicate, False otherwise.
    """
    # Simple checks first
    for existing in existing_nodes:
        # Exact name or description match
        if (
            existing.name.lower() == node.name.lower()
            or existing.description.lower() == node.description.lower()
        ):
            return True

    # Use LLM for smarter check if available
    if llm and existing_nodes:
        # Limit checks to save API calls
        nodes_to_check = existing_nodes[:5]

        # Format existing nodes for prompt
        existing_descriptions = [
            f"Name: {n.name}, Description: {n.description}" for n in nodes_to_check
        ]

        # Prompt for LLM
        duplicate_check_prompt = f"""
        Determine if the new functionality is semantically equivalent to any existing functionality.

        NEW FUNCTIONALITY:
        Name: {node.name}
        Description: {node.description}

        EXISTING FUNCTIONALITIES:
        {json.dumps(existing_descriptions, indent=2)}

        A functionality is a duplicate if it represents the SAME ACTION/CAPABILITY, even if described differently.

        Respond with ONLY "DUPLICATE" or "UNIQUE" followed by a brief explanation.
        """

        response = llm.invoke(duplicate_check_prompt)
        result = response.content.strip().upper()

        if "DUPLICATE" in result:
            return True

    return False


def validate_parent_child_relationship(parent_node, child_node, llm) -> bool:
    """
    Check if the child node makes sense as a sub-step of the parent node.

    Args:
        parent_node (FunctionalityNode): The potential parent node.
        child_node (FunctionalityNode): The potential child node.
        llm: The language model instance.

    Returns:
        bool: True if the relationship seems valid, False otherwise.
    """
    if not parent_node:
        return True  # Root nodes are always valid

    # Prompt for LLM validation
    validation_prompt = f"""
    Evaluate if the second functionality should be considered a sub-functionality of the first functionality.
    Use balanced judgment - we want to create a meaningful hierarchy without being overly strict.

    PARENT FUNCTIONALITY:
    Name: {parent_node.name}
    Description: {parent_node.description}

    POTENTIAL SUB-FUNCTIONALITY:
    Name: {child_node.name}
    Description: {child_node.description}

    A functionality should be considered a sub-functionality if it meets AT LEAST ONE of these criteria:
    1. It represents a more specific version or specialized case of the parent functionality
    2. It's normally used as part of completing the parent functionality
    3. It extends or enhances the parent functionality in a natural way
    4. It depends on the parent functionality conceptually or in workflow

    EXAMPLE VALID RELATIONSHIPS:
    - Parent: "search_products" - Child: "filter_search_results"
    - Parent: "schedule_appointment" - Child: "confirm_appointment_availability"
    - Parent: "estimate_price" - Child: "calculate_detailed_quote"
    - Parent: "manage_account" - Child: "update_profile_information"

    EXAMPLE INVALID RELATIONSHIPS:
    - Parent: "login" - Child: "view_product_catalog" (unrelated functions)
    - Parent: "check_weather" - Child: "translate_text" (completely different domains)

    Consider domain-specific logic and real-world workflows when making your determination.
    Respond with EXACTLY "VALID" or "INVALID" followed by a brief explanation.
    """

    # Get LLM response
    validation_response = llm.invoke(validation_prompt)
    result = validation_response.content.strip().upper()

    is_valid = result.startswith("VALID")

    if is_valid:
        print(
            f"  ✓ Valid relationship: '{child_node.name}' is a sub-functionality of '{parent_node.name}'"
        )
    else:
        print(
            f"  ✗ Invalid relationship: '{child_node.name}' is not related to '{parent_node.name}'"
        )

    return is_valid


def merge_similar_functionalities(
    nodes: List[FunctionalityNode], llm
) -> List[FunctionalityNode]:
    """
    Use LLM to find and merge similar nodes. Returns a new list.

    Args:
        nodes (list): List of FunctionalityNode objects to check.
        llm: The language model instance.

    Returns:
        list: A new list of FunctionalityNode objects with similar ones merged.
    """
    if not nodes or len(nodes) < 2:
        return nodes  # Nothing to merge

    result = []
    name_groups = {}

    # Group nodes by normalized name first
    for node in nodes:
        normalized_name = node.name.lower().replace("_", " ")
        if normalized_name not in name_groups:
            name_groups[normalized_name] = []
        name_groups[normalized_name].append(node)

    for name, group in name_groups.items():
        if len(group) == 1:
            # Only one node, keep it
            result.append(group[0])
            continue

        # Ask LLM if this group should be merged
        merge_prompt = f"""
        Analyze the following functionality nodes extracted from conversations with a potentially informational chatbot. Determine if they represent the **same core informational topic or achieve the same overall user goal**, even if the specific interaction steps (like providing options vs. displaying results vs. explaining) differ slightly.

        Functionality Nodes:
        {json.dumps([{"name": n.name, "description": n.description} for n in group], indent=2)}

        Consider the *purpose* and the *information conveyed*. For example, different ways of providing contact details (`display_contact_info`, `explain_contact_methods`, `repeat_contact_details`) should likely be merged into a single `provide_contact_info` node. However, `provide_contact_info` and `explain_ticketing_process` are distinct topics.

        If they represent the SAME core topic/goal, respond with exactly:
        MERGE
        name: [Suggest a concise, representative snake_case name for the core topic/goal, e.g., `provide_contact_info`]
        description: [Suggest a clear, consolidated description covering the core topic and potentially mentioning the different ways it was presented]

        If they are distinct topics or goals, respond with exactly:
        KEEP SEPARATE
        reason: [Briefly explain why they cover different topics/goals]
        """

        merge_response = llm.invoke(merge_prompt)
        content = merge_response.content.strip()

        if content.upper().startswith("MERGE"):
            # Try to parse suggested name and description
            name_match = re.search(r"name:\s*(.+)", content)
            desc_match = re.search(r"description:\s*(.+)", content)
            if name_match and desc_match:
                best_name = name_match.group(1).strip()
                best_desc = desc_match.group(1).strip()

                # Combine parameters and children from the group
                all_params = []
                merged_node = FunctionalityNode(
                    name=best_name, description=best_desc, parameters=all_params
                )

                for node in group:
                    # Merge parameters (avoid duplicates)
                    for param in node.parameters:
                        if not any(
                            p.get("name") == param.get("name") for p in all_params
                        ):
                            all_params.append(param)
                    # Add all children
                    for child in node.children:
                        merged_node.add_child(child)

                print(f"  Merged {len(group)} functionalities into '{best_name}'")
                result.append(merged_node)
            else:
                # Fallback if parsing fails: keep the first node
                print(
                    f"  WARN: Could not parse merge suggestion for group '{name}'. Keeping first node."
                )
                result.append(group[0])
        else:
            # Keep nodes separate if LLM says so
            result.extend(group)

    return result


def run_exploration_session(
    session_num,
    max_sessions,
    max_turns,
    explorer,
    the_chatbot,
    fallback_message: Optional[str] = None,
    current_node: Optional[FunctionalityNode] = None,
    explored_nodes: Optional[Set[str]] = None,
    pending_nodes: Optional[List[FunctionalityNode]] = None,
    root_nodes: Optional[List[FunctionalityNode]] = None,
    supported_languages=None,
):
    """
    Runs one chat session to explore the bot.
    Can focus on a specific 'current_node' if provided. Includes retry logic on fallback.

    Args:
        session_num (int): The current session number (0-based).
        max_sessions (int): Total sessions to run.
        max_turns (int): Max chat turns per session.
        explorer: The ChatbotExplorer instance.
        the_chatbot: The chatbot connector instance.
        fallback_message (str, optional): The detected fallback message of the chatbot. Defaults to None. # ADDED
        current_node (FunctionalityNode, optional): Node to focus exploration on. Defaults to None.
        explored_nodes (set, optional): Set of names of already explored nodes. Defaults to None.
        pending_nodes (list, optional): List of nodes waiting to be explored. Defaults to None.
        root_nodes (list, optional): List of current root nodes. Defaults to None.
        supported_languages (list, optional): List of detected languages. Defaults to None.

    Returns:
        tuple: Contains the conversation history, detected languages (None),
               new nodes found this session,
               updated root nodes list, updated pending nodes list, updated explored nodes set.
               # REMOVED fallback_message from return tuple
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
        question_response = explorer.llm.invoke(question_prompt)
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
                translation_prompt = f"Translate '{greeting_en}' to {primary_language}. Respond ONLY with the translation."
                translated_greeting = (
                    explorer.llm.invoke(translation_prompt).content.strip().strip("\"'")
                )
                # Check if translation looks okay
                if translated_greeting and len(translated_greeting.split()) > 1:
                    initial_question = translated_greeting
                else:  # Use English if translation failed
                    initial_question = greeting_en
            except Exception as e:
                print(
                    f"Warning: Failed to translate initial greeting to {primary_language}: {e}"
                )
                initial_question = greeting_en  # Fallback
        else:
            initial_question = greeting_en  # Use English

        print(
            f"   (Starting session 0 with general capability question: '{initial_question}')"
        )

    print(f"\nExplorer: {initial_question}")

    # Send first question to the chatbot
    is_ok, chatbot_message = the_chatbot.execute_with_input(initial_question)
    print(f"\nChatbot: {chatbot_message}")

    # Add first exchange to history
    conversation_history_lc.append(
        AIMessage(content=initial_question)
    )  # Explorer AI is 'assistant'
    conversation_history_lc.append(
        HumanMessage(content=chatbot_message)
    )  # Target Chatbot is 'user'/'human'

    consecutive_failures = 0
    force_topic_change_next_turn = False

    # Main loop
    turn_count = 1  # We already did the first question, so start at 1
    while True:
        # Stop if we hit the max number of turns
        if turn_count >= max_turns:
            print(
                f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}."
            )
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
            print(
                f"\n Forcing topic change: {consecutive_failures} consecutive failures. !!!"
            )
        # ---

        # --- Get what the explorer wants to say next ---
        explorer_response_content = None
        try:
            max_history_turns_for_llm = (
                10  # Keep last 10 turns (20 messages) + system prompt
            )
            messages_for_llm = [conversation_history_lc[0]] + conversation_history_lc[
                -(max_history_turns_for_llm * 2) :
            ]

            # If forcing change, add a temporary system message for this turn
            if force_topic_change_instruction:
                messages_for_llm_this_turn = messages_for_llm + [
                    SystemMessage(content=force_topic_change_instruction)
                ]
            else:
                messages_for_llm_this_turn = messages_for_llm

            # Invoke the LLM directly
            llm_response = explorer.llm.invoke(messages_for_llm_this_turn)
            explorer_response_content = llm_response.content.strip()

        except Exception as e:
            print(
                f"\nError getting response from Explorer AI LLM: {e}. Ending session."
            )
            break  # Stop if LLM fails

        if not explorer_response_content:
            print(
                "\nError: Failed to get next action from Explorer AI LLM. Ending session."
            )
            break

        print(f"\nExplorer: {explorer_response_content}")

        # If the explorer says it's done, just stop
        if "EXPLORATION COMPLETE" in explorer_response_content.upper():
            print(f"\nExplorer has finished session {session_num + 1} exploration.")
            break

        # Save what the explorer said before sending it to the chatbot
        conversation_history_lc.append(AIMessage(content=explorer_response_content))

        # --- Send the explorer's message to the chatbot ---
        is_ok, chatbot_message = the_chatbot.execute_with_input(
            explorer_response_content
        )

        if not is_ok:
            print("\nError communicating with chatbot. Ending session.")
            conversation_history_lc.append(
                HumanMessage(content="[Chatbot communication error]")
            )
            consecutive_failures += 1
            force_topic_change_next_turn = True
            break

        # Save the chatbot's first response in case we need it later
        original_chatbot_message = chatbot_message

        # Check if the chatbot gave us a fallback or error
        is_fallback = False
        if fallback_message and chatbot_message:
            # Use LLM for semantic comparison
            is_fallback = is_semantically_fallback(
                chatbot_message, fallback_message, explorer.llm
            )

        is_parsing_error = "OUTPUT_PARSING_FAILURE" in chatbot_message

        # --- Try rephrasing the message if we hit a fallback or parsing error ---
        retry_also_failed = False
        if is_fallback or is_parsing_error:
            failure_reason = (
                "Fallback message"
                if is_fallback
                else "Potential chatbot error (OUTPUT_PARSING_FAILURE)"
            )
            print(f"\n   ({failure_reason} detected. Rephrasing and retrying...)")

            # Generate a rephrased version of the original message
            rephrase_prompt = f"""
            The chatbot did not understand this message: "{explorer_response_content}"

            Please rephrase this message to convey the same intent but with different wording.
            Make the rephrased version simpler, more direct, and avoid complex structures.
            ONLY return the rephrased message, nothing else.
            """

            try:
                rephrased_response = explorer.llm.invoke(rephrase_prompt)
                rephrased_message = rephrased_response.content.strip().strip("\"'")

                if rephrased_message and rephrased_message != explorer_response_content:
                    print(f"   Original: '{explorer_response_content}'")
                    print(f"   Rephrased: '{rephrased_message}'")

                    # Try with the rephrased message
                    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(
                        rephrased_message
                    )
                else:
                    # Fallback to original if rephrasing failed or returned identical text
                    print(
                        "   Failed to generate a different rephrasing. Retrying with original."
                    )
                    is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(
                        explorer_response_content
                    )
            except Exception as e:
                print(f"   Error rephrasing message: {e}. Retrying with original.")
                is_ok_retry, chatbot_message_retry = the_chatbot.execute_with_input(
                    explorer_response_content
                )

            if is_ok_retry:
                # See if the retry gave us something different and not another failure
                is_retry_fallback = False
                if fallback_message and chatbot_message_retry:
                    is_retry_fallback = is_semantically_fallback(
                        chatbot_message_retry, fallback_message, explorer.llm
                    )

                is_retry_parsing_error = (
                    "OUTPUT_PARSING_FAILURE" in chatbot_message_retry
                )

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
            final_is_fallback = is_semantically_fallback(
                chatbot_message, fallback_message, explorer.llm
            )
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
                    f"   (Successful response this turn. Resetting consecutive failures from {consecutive_failures}.)"
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
            "role": "system"
            if isinstance(m, SystemMessage)
            else ("assistant" if isinstance(m, AIMessage) else "user"),
            "content": m.content,
        }
        for m in conversation_history_lc
    ]

    # Extract functionalities found in this session
    print("\nAnalyzing conversation for new functionalities...")
    new_functionality_nodes = extract_functionality_nodes(
        conversation_history_dict, explorer.llm, current_node
    )

    # Process newly found nodes
    if new_functionality_nodes:
        print(f"Discovered {len(new_functionality_nodes)} new functionality nodes:")

        # Merge similar nodes found *within this session* first
        new_functionality_nodes = merge_similar_functionalities(
            new_functionality_nodes, explorer.llm
        )

        for node in new_functionality_nodes:
            # Check against *all* nodes found so far
            all_existing = []
            for root in root_nodes:
                all_existing.extend(_get_all_nodes(root))  # Get all descendants

            if not is_duplicate_functionality(node, all_existing, explorer.llm):
                # If exploring a specific node, check if the new one is related
                if current_node:
                    relationship_valid = validate_parent_child_relationship(
                        current_node, node, explorer.llm
                    )

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
            root_nodes = merge_similar_functionalities(root_nodes, explorer.llm)

    # Mark the node we focused on as explored
    if current_node:
        explored_nodes.add(current_node.name)

    print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")

    # Return all updated state
    return (
        conversation_history_dict,
        None,
        new_functionality_nodes,  # Nodes found *this* session
        root_nodes,  # Updated list of roots
        pending_nodes,  # Updated pending queue
        explored_nodes,  # Updated set of explored names
    )


def _get_all_nodes(root_node):
    """
    Helper to get a flat list of nodes in a tree.

    Args:
        root_node (FunctionalityNode): The starting node of the tree/subtree.

    Returns:
        list: A flat list containing the root_node and all its descendants.
    """
    result = [root_node]
    for child in root_node.children:
        result.extend(_get_all_nodes(child))  # Recursive call
    return result


def format_conversation(messages):
    """
    Make the conversation history easy to read.

    Args:
        messages (list): The list of message dictionaries.

    Returns:
        str: A formatted string representing the conversation.
    """
    formatted = []
    for msg in messages:
        if msg["role"] in ["assistant", "user"]:
            # 'assistant' is our explorer AI, 'user' is the chatbot being tested
            sender = "Human" if msg["role"] == "assistant" else "Chatbot"
            formatted.append(f"{sender}: {msg['content']}")
    return "\n".join(formatted)
