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


def run_exploration_session(
    session_num,
    max_sessions,
    max_turns,
    explorer,
    the_chatbot,
    supported_languages=None,
    discovered_functionalities=None,
):
    """
    Run a single exploration session with the chatbot.

    Args:
        session_num: Current session number
        max_sessions: Total number of sessions
        max_turns: Maximum turns per session
        explorer: Instance of ChatbotExplorer
        the_chatbot: Chatbot connector instance
        supported_languages: List of supported languages
        discovered_functionalities: List of discovered functionalities

    Returns:
        tuple: (conversation_history, supported_languages, new_topics)
    """
    print(f"\n--- Starting Exploration Session {session_num + 1}/{max_sessions} ---")

    # Initialize empty lists if None
    if supported_languages is None:
        supported_languages = []
    if discovered_functionalities is None:
        discovered_functionalities = []

    # Add language information to the system prompt if available
    language_instruction = ""
    if session_num > 0 and supported_languages:
        language_str = ", ".join(supported_languages)
        language_instruction = f"\n\nIMPORTANT: The chatbot supports these languages: {language_str}. Adapt your questions accordingly and consider using these languages in your exploration."

    primary_language = (
        supported_languages[0]
        if supported_languages and len(supported_languages) > 0
        else "English"
    )

    total_sessions = max_sessions
    # We divide things into 5 phases
    exploration_phase = (session_num * 5) // total_sessions

    # Build context from previous discoveries
    discovery_context = ""
    if session_num > 0 and discovered_functionalities:
        # Limit to 5 most recent discoveries to keep prompts manageable
        recent_discoveries = discovered_functionalities[-5:]
        discovery_context = f"\n\nPreviously discovered features/topics: {', '.join(recent_discoveries)}"

    # Define session focus based on exploration phase
    if exploration_phase == 0:
        # First 20% Basic information and capabilities
        session_focus = (
            "Explore basic information and general capabilities of the chatbot"
        )
    elif exploration_phase == 1:
        # Next 20% General services building on what we know
        session_focus = f"Investigate general services and features of the chatbot{discovery_context}"
    elif exploration_phase == 2:
        # Next 20% Dig deeper into specific features
        session_focus = f"Explore more details about specific features we've discovered{discovery_context}"
    elif exploration_phase == 3:
        # Next 20% Advanced usage of discovered features
        session_focus = f"Investigate advanced usage scenarios for these features{discovery_context}"
    else:
        # Final 20% Mix of new discovery and some edge cases
        if session_num % 2 == 0:
            session_focus = (
                "Discover any additional features or services we might have missed"
            )
        else:
            session_focus = (
                f"Test specific scenarios related to these features{discovery_context}"
            )

    # Create the system prompt
    system_content = f"""You are an Explorer AI tasked with learning about another chatbot you're interacting with.

    IMPORTANT GUIDELINES:
    1. Ask ONE simple question at a time - the chatbot gets confused by multiple questions
    2. Keep your messages short and direct
    3. When the chatbot indicates it didn't understand, simplify your language further
    4. Follow the chatbot's conversation flow and adapt to its capabilities{language_instruction}

    EXPLORATION FOCUS FOR SESSION {session_num + 1}:
    {session_focus}

    Your goal is to understand the chatbot's capabilities through direct, simple interactions.
    After {max_turns} exchanges, or when you feel you've explored this path thoroughly, say "EXPLORATION COMPLETE".
    """

    # Reset conversation history for this session since each session is a new conversation
    conversation_history = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    print("Starting session")

    # Generate the initial question
    initial_question = ""
    if session_num == 0:
        initial_question = "Hello! What languages do you support or speak?"
    else:
        language_str = ", ".join(supported_languages)

        # Create a prompt for the Explorer to generate an initial question
        question_prompt = f"""
        You need to generate an initial question for a conversation with a chatbot.

        INFORMATION:
        - This is session {session_num + 1} of the exploration
        - The chatbot supports these languages: {language_str}
        - Primary language: {primary_language}
        {discovery_context}

        EXPLORATION FOCUS FOR THIS SESSION:
        {session_focus}

        IMPORTANT:
        - Keep your question simple and direct - only ask ONE thing
        - Your response should only contain the question, nothing else
        - AVOID simply asking about limitations
        - If we have previous discoveries, build upon them rather than repeating basic questions
        - Make your question specific enough to learn new information

        GENERATE A SIMPLE OPENING QUESTION THAT:
        1. Is appropriate for starting this exploration session
        2. Is in the primary supported language of the chatbot ({primary_language})
        3. Helps discover more details about chatbot capabilities
        4. Follows a logical exploration path based on what we already know
        """

        # Generate the initial question using the Explorer's LLM
        question_response = explorer.llm.invoke(question_prompt)
        initial_question = question_response.content.strip().strip("\"'")

    # Start with our question instead of waiting for chatbot to start
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
        # Here only the explorer node will get executed since exploration finished is False
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

    # Extract supported languages if this is the first session
    new_supported_languages = None
    if session_num == 0:
        new_supported_languages = extract_supported_languages(
            chatbot_message, explorer.llm
        )
        print(f"\nDetected supported languages: {new_supported_languages}")

    # Extract key topics discovered in this session
    def format_conversation(messages):
        """Format conversation history into a readable string"""
        formatted = []
        for msg in messages:
            if msg["role"] in ["assistant", "user"]:
                # The explorer is the "Human" and the chatbot is "Chatbot"
                sender = "Human" if msg["role"] == "assistant" else "Chatbot"
                formatted.append(f"{sender}: {msg['content']}")
        return "\n".join(formatted)

    session_topics_prompt = f"""
    Review this conversation and identify 2-3 key features or capabilities of the chatbot that were discovered.
    List ONLY the features as short phrases (3-5 words each). Don't include explanations or commentary.

    CONVERSATION:
    {format_conversation(conversation_history)}

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    FEATURE: [feature 1]
    FEATURE: [feature 2]
    FEATURE: [feature 3]
    """

    topics_response = explorer.llm.invoke(session_topics_prompt)

    # Simple extraction of features using the structured format
    new_topics = []
    for line in topics_response.content.strip().split("\n"):
        if line.startswith("FEATURE:"):
            topic = line[len("FEATURE:") :].strip()
            if topic and len(topic) > 3:
                new_topics.append(topic)

    print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")

    return conversation_history, new_supported_languages, new_topics
