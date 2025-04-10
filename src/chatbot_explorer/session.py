from typing import List, Dict, Any, Optional, Tuple, Set
from .functionality_node import FunctionalityNode
import re


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

    Extract the distinct functionalities/actions the chatbot can perform based on this conversation.
    For each functionality, identify:
    1. A short name (use snake_case)
    2. A clear description
    3. Required parameters (if any)

    Format each functionality EXACTLY as:
    FUNCTIONALITY:
    name: snake_case_name
    description: Clear description of what this functionality does
    parameters: param1, param2 (or "None" if no parameters)

    If you identify multiple functionalities, list each one separately with the FUNCTIONALITY: header.
    If no new functionality is identified, respond with "NO_NEW_FUNCTIONALITY".
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

    return functionality_nodes


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

    Extract the distinct functionalities/actions the chatbot can perform based on this conversation.
    For each functionality, identify:
    1. A short name (use snake_case)
    2. A clear description
    3. Required parameters (if any)

    Format each functionality EXACTLY as:
    FUNCTIONALITY:
    name: snake_case_name
    description: Clear description of what this functionality does
    parameters: param1, param2 (or "None" if no parameters)

    If you identify multiple functionalities, list each one separately with the FUNCTIONALITY: header.
    If no new functionality is identified, respond with "NO_NEW_FUNCTIONALITY".
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

    return functionality_nodes


def is_duplicate_functionality(
    node: FunctionalityNode, existing_nodes: List[FunctionalityNode]
) -> bool:
    """
    Check if a node represents functionality that's already captured in existing nodes.

    This uses semantic similarity rather than exact name matching.
    """
    for existing in existing_nodes:
        # For now check if description or name are the same
        if (
            existing.description.lower() == node.description.lower()
            or existing.name.lower() == node.name.lower()
        ):
            return True

        # TODO: Ask LLM for similarity check

    return False


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

    # Add language information to the system prompt if available
    language_instruction = ""
    if session_num > 0 and supported_languages:
        language_str = ", ".join(supported_languages)
        language_instruction = f"\n\nIMPORTANT: The chatbot supports these languages: {language_str}. Adapt your questions accordingly."

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

    # Generate the initial question based on current node context
    if current_node:
        question_prompt = f"""
        You need to generate an initial question to explore a chatbot functionality.

        FUNCTIONALITY TO EXPLORE:
        Name: {current_node.name}
        Description: {current_node.description}
        Parameters: {", ".join(p.get("name", "?") for p in current_node.parameters) if current_node.parameters else "None"}

        Generate a simple, direct question that would help explore this functionality in depth.
        Your question should be appropriate for starting a conversation about this specific feature.
        """

        question_response = explorer.llm.invoke(question_prompt)
        initial_question = question_response.content.strip().strip("\"'")
    else:
        initial_question = "Hello! What can you help me with?"

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
        for node in new_functionality_nodes:
            # Check for duplicates against all existing nodes
            all_existing = []
            for root in root_nodes:
                all_existing.extend(_get_all_nodes(root))

            if not is_duplicate_functionality(node, all_existing):
                if current_node:
                    # Add as child to current node
                    current_node.add_child(node)
                    print(f"  - '{node.name}' (child of '{current_node.name}')")
                else:
                    # Add as root node
                    root_nodes.append(node)
                    print(f"  - '{node.name}' (root node)")

                # Add to pending nodes for exploration
                pending_nodes.append(node)
            else:
                print(f"  - Skipped duplicate functionality: '{node.name}'")
    else:
        print("No new functionalities discovered in this session.")

    # Mark current node as explored if we're exploring one
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
