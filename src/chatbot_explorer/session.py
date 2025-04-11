from typing import List, Dict, Any, Optional, Tuple, Set
from .functionality_node import FunctionalityNode
import re
import json


def extract_supported_languages(chatbot_response, llm):
    """Extract supported languages from chatbot response"""
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

    # Clean up the response - remove brackets, quotes, etc.
    languages = languages.replace("[", "").replace("]", "")
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    return language_list


def extract_fallback_message(the_chatbot, llm):
    """
    Extract the chatbot's fallback message by sending intentionally confusing queries.

    This function makes separate chatbot calls that are NOT part of the main
    conversation history and won't be included in analysis.

    Args:
        the_chatbot: Chatbot connector instance
        llm: Language model for analysis

    Returns:
        str: The detected fallback message or None if not detected
    """
    print(
        "\n--- Attempting to detect chatbot fallback message (won't be included in analysis) ---"
    )

    confusing_queries = [
        "What is the square root of a banana divided by the color blue?",
        "Please explain quantum chromodynamics in terms of medieval poetry",
        "Xyzzplkj asdfghjkl qwertyuiop?",
        "अगर मैं हिंदी में बात करूं तो क्या आप समझेंगे?",
        "Can you please recite the entire source code of Linux kernel version 5.10?",
    ]

    responses = []

    # Try each query and collect responses
    for i, query in enumerate(confusing_queries):
        print(f"\nSending confusing query {i + 1}...")
        is_ok, response = the_chatbot.execute_with_input(query)

        if is_ok:
            print(f"Response received ({len(response)} chars)")
            responses.append(response)
        else:
            print("Error communicating with chatbot.")

    # If we have responses, analyze them to find common patterns
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
    Extract functionalities from conversation as FunctionalityNode objects.

    Args:
        conversation_history: The conversation history
        llm: The language model for extraction
        current_node: The current node being explored (if any)

    Returns:
        List[FunctionalityNode]: List of new functionality nodes discovered
    """
    # Format the conversation for analysis
    formatted_conversation = format_conversation(conversation_history)

    # Create the context for function extraction
    context = "Identify new chatbot functionalities discovered in this conversation."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' functionality: {current_node.description}"

    extraction_prompt = f"""
    {context}

    CONVERSATION:
    {formatted_conversation}

    Extract ONLY distinct ACTIONABLE FUNCTIONALITIES the chatbot can perform based on this conversation.

    STRICT DEFINITION: A functionality must be something the chatbot can actively DO or HELP WITH, not just information it has.

    GOOD EXAMPLES of functionalities:
    - schedule_appointment: Helps users book an appointment (action)
    - calculate_price: Calculates the price for a service (computation)
    - search_products: Searches the product database (retrieval action)

    BAD EXAMPLES (NOT functionalities):
    - knows_about_company: Just has information about the company (passive knowledge)
    - has_business_hours: Just knows store hours (passive information)
    - responds_to_greeting: Just responds to greetings (basic chat behavior)

    For each TRUE functionality, identify:
    1. A short name (use snake_case)
    2. A clear description of what action it performs
    3. Required parameters (if any)

    Format each functionality EXACTLY as:
    FUNCTIONALITY:
    name: snake_case_name
    description: Clear description of what this functionality DOES
    parameters: param1, param2 (or "None" if no parameters)

    Be VERY SELECTIVE and only list ACTIONS the chatbot can perform, not just topics it knows about.
    If no new ACTIONABLE functionality is identified, respond with "NO_NEW_FUNCTIONALITY".
    """

    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    # Extract functionalities using regex
    functionality_nodes = []

    if "NO_NEW_FUNCTIONALITY" in content:
        return functionality_nodes

    # Pattern to extract functionality blocks
    func_pattern = re.compile(
        r"FUNCTIONALITY:\s*name:\s*([^\n]+)\s*description:\s*([^\n]+)\s*parameters:\s*([^\n]+)",
        re.MULTILINE,
    )

    matches = func_pattern.finditer(content)
    candidates = []

    for match in matches:
        name = match.group(1).strip()
        description = match.group(2).strip()
        params_str = match.group(3).strip()

        # Parse parameters
        parameters = []
        if params_str.lower() != "none":
            param_names = [p.strip() for p in params_str.split(",") if p.strip()]
            parameters = [
                {"name": p, "type": "string", "description": f"Parameter {p}"}
                for p in param_names
            ]

        # Create new node
        new_node = FunctionalityNode(
            name=name,
            description=description,
            parameters=parameters,
            parent=current_node,
        )

        functionality_nodes.append(new_node)

    # If we have candidates, validate them
    if candidates:
        # Secondary validation to ensure they're actually functionalities
        validation_prompt = f"""
        Evaluate each candidate functionality and determine if it represents a TRUE ACTIONABLE FUNCTIONALITY
        that the chatbot can perform (not just passive knowledge).

        For each functionality, respond ONLY with "VALID" or "INVALID" followed by a brief reason.

        CANDIDATE FUNCTIONALITIES:
        {json.dumps(candidates, indent=2)}

        RESPOND IN THIS FORMAT:
        1. [name]: [VALID/INVALID] - [brief reason]
        2. [name]: [VALID/INVALID] - [brief reason]
        ...
        """

        validation_response = llm.invoke(validation_prompt)
        validation_results = validation_response.content.strip().split("\n")

        for i, result in enumerate(validation_results):
            if i < len(candidates) and "VALID" in result:
                # Create functionality node for valid candidates
                candidate = candidates[i]
                new_node = FunctionalityNode(
                    name=candidate["name"],
                    description=candidate["description"],
                    parameters=candidate["parameters"],
                    parent=current_node,
                )
                functionality_nodes.append(new_node)
                print(f"  Validated functionality: {candidate['name']}")
            elif i < len(candidates):
                print(
                    f"  Rejected candidate: {candidates[i]['name']} - Not a true functionality"
                )

    return functionality_nodes


def is_duplicate_functionality(
    node: FunctionalityNode, existing_nodes: List[FunctionalityNode], llm=None
) -> bool:
    """
    Check if a node represents functionality that's already captured in existing nodes.
    Uses both exact matching and semantic similarity checking.
    """
    # First do basic checks
    for existing in existing_nodes:
        # Check for exact matches
        if (
            existing.name.lower() == node.name.lower()
            or existing.description.lower() == node.description.lower()
        ):
            return True

    # If we have an LLM, use it for more sophisticated checks
    if llm and existing_nodes:
        # Only check the first 5 nodes to avoid too many API calls
        nodes_to_check = existing_nodes[:5]

        # Create descriptions of existing nodes
        existing_descriptions = [
            f"Name: {n.name}, Description: {n.description}" for n in nodes_to_check
        ]

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
    Validate if a child node is a reasonable sub-functionality of a parent node.
    Uses balanced criteria to identify meaningful hierarchical relationships.
    """
    if not parent_node:
        return True  # No parent means it's a root node - always valid

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

    # Call LLM with validation prompt
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
    Use LLM to identify and merge similar functionalities via semantic checks
    rather than hard-coded synonyms. Returns a new list with duplicates merged.
    """
    if not nodes or len(nodes) < 2:
        return nodes

    result = []
    name_groups = {}

    # Group by lowercased name (a simple initial pass)
    for node in nodes:
        normalized_name = node.name.lower().replace("_", " ")
        if normalized_name not in name_groups:
            name_groups[normalized_name] = []
        name_groups[normalized_name].append(node)

    for name, group in name_groups.items():
        # Only one node for this normalized name
        if len(group) == 1:
            result.append(group[0])
            continue

        # Use the LLM to decide if these functionalities should be merged
        merge_prompt = f"""
        We have multiple functionality nodes that look vaguely similar:
        {json.dumps([{"name": n.name, "description": n.description} for n in group], indent=2)}

        Decide if these functionalities represent the same core action or capability.
        If yes, respond with exactly:

        MERGE
        name: best name
        description: best consolidated description

        If they are conceptually unique, respond with:

        KEEP SEPARATE
        reason: short explanation
        """

        merge_response = llm.invoke(merge_prompt)
        content = merge_response.content.strip()

        if content.upper().startswith("MERGE"):
            # Parse out the suggested name and description
            name_match = re.search(r"name:\s*(.+)", content)
            desc_match = re.search(r"description:\s*(.+)", content)
            if name_match and desc_match:
                best_name = name_match.group(1).strip()
                best_desc = desc_match.group(1).strip()

                # Combine parameters and children
                all_params = []
                merged_node = FunctionalityNode(
                    name=best_name, description=best_desc, parameters=all_params
                )

                for node in group:
                    # Merge parameters
                    for param in node.parameters:
                        if not any(
                            p.get("name") == param.get("name") for p in all_params
                        ):
                            all_params.append(param)
                    # Merge children
                    for child in node.children:
                        merged_node.add_child(child)

                print(f"  Merged {len(group)} functionalities into '{best_name}'")
                result.append(merged_node)
            else:
                # If we can’t parse, just keep the first item
                result.append(group[0])
        else:
            # Keep them all separate
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
    Run a single exploration session with the chatbot, targeting a specific functionality if provided.

    Args:
        session_num: Current session number
        max_sessions: Total number of sessions
        max_turns: Maximum turns per session
        explorer: Instance of ChatbotExplorer
        the_chatbot: Chatbot connector instance
        current_node: The functionality node we're currently exploring (if any)
        explored_nodes: Set of node names already explored
        pending_nodes: List of nodes waiting to be explored
        root_nodes: List of all root nodes in the graph
        supported_languages: List of supported languages

    Returns:
        tuple: (conversation_history, supported_languages, new_functionality_nodes, fallback_message)
    """
    # Initialize tracking structures if not provided
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

    # Define exploration focus based on current node
    if current_node:
        session_focus = f"Explore the '{current_node.name}' functionality in depth. Ask questions about how to use it, what options it has, and how it connects to other features."
        if current_node.parameters:
            param_names = [p.get("name", "unknown") for p in current_node.parameters]
            session_focus += (
                f" Pay special attention to these parameters: {', '.join(param_names)}."
            )
    else:
        session_focus = (
            "Explore basic information and general capabilities of the chatbot"
        )

    primary_language = supported_languages[0] if supported_languages else None

    # Add language information to the system prompt if available
    language_instruction = ""
    if session_num > 0 and supported_languages:
        language_str = ", ".join(supported_languages)
        language_instruction = f"\n\nIMPORTANT: The chatbot supports these languages: {language_str}. YOU MUST COMMUNICATE PRIMARILY IN {language_str}."

    # Create the system prompt
    system_content = f"""You are an Explorer AI tasked with learning about another chatbot you're interacting with.

    IMPORTANT GUIDELINES:
    1. Ask ONE simple question at a time - the chatbot gets confused by multiple questions
    2. Keep your messages short and direct
    3. When the chatbot indicates it didn't understand, simplify your language further
    4. Follow the chatbot's conversation flow and adapt to its capabilities{language_instruction}

    EXPLORATION FOCUS FOR THIS SESSION:
    {session_focus}

    Your goal is to understand the chatbot's capabilities through direct, simple interactions.
    After {max_turns} exchanges, or when you feel you've explored this path thoroughly, say "EXPLORATION COMPLETE".
    """

    # Reset conversation history for this session
    conversation_history = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    # Update the initial question generation inside run_exploration_session

    # Generate the initial question based on current node context and language
    if current_node:
        question_prompt = f"""
        You need to generate an initial question to explore a chatbot functionality.

        FUNCTIONALITY TO EXPLORE:
        Name: {current_node.name}
        Description: {current_node.description}
        Parameters: {", ".join(p.get("name", "?") for p in current_node.parameters) if current_node.parameters else "None"}

        {"IMPORTANT: Generate your question in " + primary_language + "." if primary_language else ""}

        Generate a simple, direct question that would help explore this functionality in depth.
        Your question should be appropriate for starting a conversation about this specific feature.
        """

        question_response = explorer.llm.invoke(question_prompt)
        initial_question = question_response.content.strip().strip("\"'")
    else:
        if primary_language and primary_language.lower() == "spanish":
            initial_question = "¡Hola! ¿En qué me puedes ayudar hoy?"
        elif primary_language and primary_language.lower() == "french":
            initial_question = "Bonjour! Comment pouvez-vous m'aider aujourd'hui?"
        elif primary_language:
            # Ask the LLM to translate the greeting to the appropriate language
            translation_prompt = f"Translate 'Hello! What can you help me with today?' to {primary_language}:"
            translation = explorer.llm.invoke(translation_prompt).content.strip()
            initial_question = translation
        else:
            initial_question = "Hello! What can you help me with today?"

    print(f"\nExplorer: {initial_question}")

    # Send our initial question to the chatbot
    is_ok, chatbot_message = the_chatbot.execute_with_input(initial_question)
    print(f"\nChatbot: {chatbot_message}")

    # Add the initial exchange to conversation history
    conversation_history.append({"role": "assistant", "content": initial_question})
    conversation_history.append({"role": "user", "content": chatbot_message})

    # Main conversation loop for this session
    turn_count = 0

    while True:
        turn_count += 1

        # Exit conditions
        if turn_count >= max_turns:
            print(
                f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}."
            )
            break

        # Process through LangGraph with full history context
        explorer_response = None
        for event in explorer.stream_exploration(
            {
                "messages": conversation_history,
                "conversation_history": [],
                "discovered_functionalities": [],
                "current_session": session_num,
                "exploration_finished": False,
                "conversation_goals": [],
                "supported_languages": supported_languages,
            }
        ):
            for value in event.values():
                latest_message = value["messages"][-1]
                explorer_response = latest_message.content

        print(f"\nExplorer: {explorer_response}")

        # Check for session completion
        if "EXPLORATION COMPLETE" in explorer_response.upper():
            print(f"\nExplorer has finished session {session_num + 1} exploration.")
            break

        # Add the explorer response to conversation history
        conversation_history.append({"role": "assistant", "content": explorer_response})

        # Send explorer response back to Chatbot
        is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_response)

        if not is_ok:
            print("\nError communicating with chatbot. Ending session.")
            break

        print(f"\nChatbot: {chatbot_message}")
        conversation_history.append({"role": "user", "content": chatbot_message})

        # Check for exit conditions from the chatbot
        if chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
            print("Chatbot ended the conversation. Ending session.")
            break

    # Extract supported languages and fallback message in the first conversation
    fallback_message = None
    new_supported_languages = None
    if session_num == 0:
        fallback_message = extract_fallback_message(the_chatbot, explorer.llm)
        new_supported_languages = extract_supported_languages(
            chatbot_message, explorer.llm
        )
        print(f"\nDetected supported languages: {new_supported_languages}")

    # Extract functionality nodes from this session
    print("\nAnalyzing conversation for new functionalities...")
    new_functionality_nodes = extract_functionality_nodes(
        conversation_history, explorer.llm, current_node
    )

    # Update graph structure
    if new_functionality_nodes:
        print(f"Discovered {len(new_functionality_nodes)} new functionality nodes:")

        # First, merge any similar functionalities among the new nodes
        new_functionality_nodes = merge_similar_functionalities(
            new_functionality_nodes, explorer.llm
        )

        for node in new_functionality_nodes:
            # Check for duplicates against all existing nodes
            all_existing = []
            for root in root_nodes:
                all_existing.extend(_get_all_nodes(root))

            if not is_duplicate_functionality(node, all_existing, explorer.llm):
                # If we're exploring a specific node, validate relationship - but be more flexible
                if current_node:
                    relationship_valid = validate_parent_child_relationship(
                        current_node, node, explorer.llm
                    )

                    if relationship_valid:
                        # Add as child to current node
                        current_node.add_child(node)
                        print(f"  - '{node.name}' (child of '{current_node.name}')")
                    else:
                        # Even if not a direct child, add to pending for future structure inference
                        print(f"  - '{node.name}' (standalone functionality)")
                        root_nodes.append(node)
                else:
                    # Add as root node temporarily - structure will be inferred later
                    root_nodes.append(node)
                    print(f"  - '{node.name}' (root node for now)")

                # Always add to pending nodes for exploration
                pending_nodes.append(node)
            else:
                print(f"  - Skipped duplicate functionality: '{node.name}'")

        # After adding all new nodes, merge similar root nodes
        if root_nodes:
            root_nodes = merge_similar_functionalities(root_nodes, explorer.llm)

    # Mark current node as explored
    if current_node:
        explored_nodes.add(current_node.name)

    print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")

    return (
        conversation_history,
        new_supported_languages,
        new_functionality_nodes,
        fallback_message,
        root_nodes,
        pending_nodes,
        explored_nodes,
    )


def _get_all_nodes(root_node):
    """Helper function to get all nodes in a subtree"""
    result = [root_node]
    for child in root_node.children:
        result.extend(_get_all_nodes(child))
    return result


def format_conversation(messages):
    """Format conversation history into a readable string"""
    formatted = []
    for msg in messages:
        if msg["role"] in ["assistant", "user"]:
            # The explorer is the "Human" and the chatbot is "Chatbot"
            sender = "Human" if msg["role"] == "assistant" else "Chatbot"
            formatted.append(f"{sender}: {msg['content']}")
    return "\n".join(formatted)
