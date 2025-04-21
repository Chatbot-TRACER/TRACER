from typing import List, Dict, Any, Optional, Tuple, Set
from .functionality_node import FunctionalityNode
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
        "अगर मैं हिंदी में बात करूं तो क्या आप समझेंगे?",
        "Can you please recite the entire source code of Linux kernel version 5.10?",
    ]

    responses = []

    # Send confusing queries and get responses
    for i, query in enumerate(confusing_queries):
        print(f"\nSending confusing query {i + 1}...")
        is_ok, response = the_chatbot.execute_with_input(query)

        if is_ok:
            print(f"Response received ({len(response)} chars)")
            responses.append(response)
        else:
            print("Error communicating with chatbot.")

    # Analyze responses if we got any
    if responses:
        analysis_prompt = f"""
        I'm trying to identify a chatbot's fallback message - the standard response it gives when it doesn't understand.

        Below are responses to intentionally confusing or nonsensical questions.
        If there's a consistent pattern or identical response, that's likely the fallback message.

        RESPONSES:
        {responses}

        1. Is there an identical or very similar response pattern? If so, extract it exactly.
        2. If responses vary but have a common theme or structure, describe that pattern.
        3. If there's no clear pattern, select the response that seems most like a generic fallback.

        RETURN ONLY THE EXTRACTED FALLBACK MESSAGE OR PATTERN, NOTHING ELSE:
        """

        fallback_result = llm.invoke(analysis_prompt)
        fallback = fallback_result.content.strip()

        print(
            f'Detected fallback message: "{fallback[:50]}{"..." if len(fallback) > 50 else ""}"'
        )
        return fallback

    print("Could not detect a consistent fallback message.")
    return None


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

    Analyze the conversation and extract distinct **interaction steps, user goals, or informational topics**. Identify the most meaningful representation of what happened.

    - For **transactional bots** (guiding through steps): Focus on the specific ACTIONS taken by the user or chatbot (e.g., `select_pizza_size`, `add_item_to_cart`, `confirm_order`).
    - For **informational bots** (answering questions): Focus on the TOPIC of information provided (e.g., `provide_contact_info`, `explain_wifi_setup`).
    - Use clear, descriptive snake_case names.

    AVOID extracting functionalities that solely describe the chatbot failing to understand or provide information (e.g., 'handle_repeat_requests', 'fail_to_provide_info'). Focus on successful actions or information provided, or capabilities the chatbot *claims* to have.

    For each relevant step/topic, identify:
    1. A short, descriptive name (snake_case).
    2. A clear description.
    3. Required parameters (inputs needed, comma-separated or "None").

    Format EXACTLY as:
    FUNCTIONALITY:
    name: snake_case_name
    description: Clear description.
    parameters: param1, param2 (or "None")

    If no new relevant step/topic is identified, respond ONLY with "NO_NEW_FUNCTIONALITY".
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
    current_node: Optional[FunctionalityNode] = None,
    explored_nodes: Optional[Set[str]] = None,
    pending_nodes: Optional[List[FunctionalityNode]] = None,
    root_nodes: Optional[List[FunctionalityNode]] = None,
    supported_languages=None,
):
    """
    Runs one chat session to explore the bot.
    Can focus on a specific 'current_node' if provided.

    Args:
        session_num (int): The current session number (0-based).
        max_sessions (int): Total sessions to run.
        max_turns (int): Max chat turns per session.
        explorer: The ChatbotExplorer instance.
        the_chatbot: The chatbot connector instance.
        current_node (FunctionalityNode, optional): Node to focus exploration on. Defaults to None.
        explored_nodes (set, optional): Set of names of already explored nodes. Defaults to None.
        pending_nodes (list, optional): List of nodes waiting to be explored. Defaults to None.
        root_nodes (list, optional): List of current root nodes. Defaults to None.
        supported_languages (list, optional): List of detected languages. Defaults to None.

    Returns:
        tuple: Contains the conversation history, detected languages (likely None),
               new nodes found this session, detected fallback (likely None),
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
    4. **ADAPTIVE EXPLORATION:**
        - **If the chatbot provides information (like an explanation, contact details, status update) OR a fallback/error message, and does NOT ask a question or offer clear interactive choices:**
            a) **Check for Repetitive Failure:** If the chatbot has given the **exact same fallback/error message** for the last 2-3 turns despite you asking different relevant questions about the *same topic*, **ABANDON this topic for this session**. Your next turn should be to ask about a **completely different capability** or topic you know exists or is plausible (e.g., switch from asking about custom pizza ingredients to asking about predefined pizzas or drinks), OR if no other path is obvious, respond with "EXPLORATION COMPLETE".
            b) **If not repetitive failure:** Ask a specific, relevant clarifying question about the information/fallback provided OR ask about a NEW, specific, plausible topic/task relevant to the chatbot's likely domain (infer this domain). Do NOT just ask "What else?".
        - **Otherwise (if the bot asks a question or offers choices):** Respond appropriately to continue the current flow or make a selection as per Guideline 3.
    5. Prioritize actions/questions relevant to the `EXPLORATION FOCUS` below.
    6. Follow the chatbot's conversation flow naturally. {language_instruction}

    EXPLORATION FOCUS FOR THIS SESSION:
    {session_focus}

    Try to follow the focus and the adaptive exploration guideline, especially the rule about abandoning topics after repetitive failures. After {max_turns} exchanges, or when you believe you have thoroughly explored this specific path/topic (or reached a dead end/loop), respond ONLY with "EXPLORATION COMPLETE".
    """

    # Start fresh conversation history
    conversation_history = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

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
            "What are the main things you can do?",
            "What topics can you provide information about?",
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
    conversation_history.append({"role": "assistant", "content": initial_question})
    conversation_history.append({"role": "user", "content": chatbot_message})

    # Main chat loop
    turn_count = 0
    while True:
        turn_count += 1

        # Stop if max turns reached
        if turn_count >= max_turns:
            print(
                f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}."
            )
            break

        # Get explorer's next response using LangGraph
        explorer_response = None
        for event in explorer.stream_exploration(
            {
                "messages": conversation_history,
                "conversation_history": [],  # Not needed here?
                "discovered_functionalities": [],  # Not needed here?
                "current_session": session_num,
                "exploration_finished": False,
                "conversation_goals": [],  # Not used?
                "supported_languages": supported_languages,
            }
        ):
            for value in event.values():
                # Get the last message added by the graph
                latest_message = value["messages"][-1]
                explorer_response = latest_message.content

        print(f"\nExplorer: {explorer_response}")

        # Stop if explorer says it's done
        if "EXPLORATION COMPLETE" in explorer_response.upper():
            print(f"\nExplorer has finished session {session_num + 1} exploration.")
            break

        # Add explorer response to history
        conversation_history.append({"role": "assistant", "content": explorer_response})

        # Send explorer response to the chatbot
        is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_response)

        if not is_ok:
            print("\nError communicating with chatbot. Ending session.")
            break

        print(f"\nChatbot: {chatbot_message}")
        # Add chatbot response to history
        conversation_history.append({"role": "user", "content": chatbot_message})

        # Stop if chatbot says goodbye
        if chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
            print("Chatbot ended the conversation. Ending session.")
            break

    # After the loop, do analysis for the first session
    fallback_message = None
    new_supported_languages = None  # Language detection moved to main.py pre-probe
    if session_num == 0:
        # Try to find the fallback message
        fallback_message = extract_fallback_message(the_chatbot, explorer.llm)
        # Language detection is now done before the loop starts

    # Extract functionalities found in this session
    print("\nAnalyzing conversation for new functionalities...")
    new_functionality_nodes = extract_functionality_nodes(
        conversation_history, explorer.llm, current_node
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

                # Add all non-duplicate new nodes to the pending queue
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
        conversation_history,
        new_supported_languages,  # Will be None unless session_num == 0 (and even then, likely None now)
        new_functionality_nodes,  # Nodes found *this* session
        fallback_message,  # Fallback found *this* session (only if session_num == 0)
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
