import sys
import yaml
import os
import re
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer, extract_supported_languages
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam

# Takes anything that is between exactly two curly braces
VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def main():
    # Parse command line arguments
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Use the parameters from args
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_file = args.output
    technology = args.technology

    # Validate the technology argument
    if args.technology not in valid_technologies:
        print(
            f"Invalid technology: {args.technology}. Must be one of: {valid_technologies}"
        )
        sys.exit(1)

    # Display configuration
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {args.technology}")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print(f"Output file: {args.output}")
    print("====================================")

    # Initialize explorer
    explorer = ChatbotExplorer(model_name)

    # Track multiple conversation sessions
    conversation_sessions = []
    supported_languages = []

    # Create the chatbot according to the technology
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)

    # We store them to try and expand
    discovered_functionalities = []

    # Session management loop
    for session_num in range(max_sessions):
        print(
            f"\n--- Starting Exploration Session {session_num + 1}/{max_sessions} ---"
        )

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
        # We can divide things into 5 phases
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
                session_focus = f"Test specific scenarios related to these features{discovery_context}"

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

        # Reset conversation history for this session
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

            # Add the chatbot message to the conversation history
            conversation_history.append({"role": "user", "content": chatbot_message})

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
            conversation_history.append(
                {"role": "assistant", "content": explorer_response}
            )

            # Send explorer response back to Chatbot
            is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_response)

            if not is_ok:
                print("\nError communicating with chatbot. Ending session.")
                break

            print(f"\nChatbot: {chatbot_message}")

            # Check for exit conditions from the chatbot
            if chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
                print("Chatbot ended the conversation. Ending session.")
                break

        # At the end of the first session get the supported languages
        if session_num == 0:
            supported_languages = extract_supported_languages(
                chatbot_message, explorer.llm
            )
            print(f"\nDetected supported languages: {supported_languages}")

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

        # Add unique topics to our tracking
        for topic in new_topics:
            if topic not in discovered_functionalities:
                discovered_functionalities.append(topic)
                print(f"New functionality discovered: {topic}")

        # After session ends, save the conversation history
        print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")
        conversation_sessions.append(conversation_history)

    # Create state for analysis
    print("\n--- All exploration sessions complete. Analyzing results... ---")
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            }
        ],
        "conversation_history": conversation_sessions,
        "discovered_functionalities": [],
        "discovered_limitations": [],
        "current_session": max_sessions,
        "exploration_finished": True,
        "conversation_goals": [],
        "supported_languages": supported_languages,
    }

    # Execute the analysis
    config = {"configurable": {"thread_id": "analysis_session"}}
    result = explorer.run_exploration(analysis_state, config)

    # Display results with error handling for the missing key
    print("\n=== CHATBOT FUNCTIONALITY ANALYSIS ===")
    print("\n## FUNCTIONALITIES")
    for i, func in enumerate(result.get("discovered_functionalities", []), 1):
        print(f"{i}. {func}")

    print("\n## LIMITATIONS")
    if "discovered_limitations" in result:
        for i, limitation in enumerate(result["discovered_limitations"], 1):
            print(f"{i}. {limitation}")
    else:
        print("No limitations discovered.")

    # Save results with error handling
    with open(output_file, "w") as f:
        f.write("## FUNCTIONALITIES\n")
        for func in result.get("discovered_functionalities", []):
            f.write(f"- {func}\n")
        f.write("\n## LIMITATIONS\n")
        if "discovered_limitations" in result:
            for limitation in result["discovered_limitations"]:
                f.write(f"- {limitation}\n")
        else:
            f.write("- No limitations discovered.\n")

    # Generate user profiles and goals
    print("\n--- User profiles and goals from analysis ---")
    profiles_list = result.get("conversation_goals", [])
    supported_languages = result.get("supported_languages", [])

    primary_language = supported_languages[0] if supported_languages else "English"

    if profiles_list:
        output_dir = "profiles"
        os.makedirs(output_dir, exist_ok=True)
        for profile in profiles_list:
            # Obtain the variables in the profile
            used_variables = set()
            for goal in profile.get("goals", []):
                variables_in_goals = VARIABLE_PATTERN.findall(goal)
                used_variables.update(variables_in_goals)

            yaml_goals = []

            # Add the goals to the yaml
            for goal in profile.get("goals", []):
                yaml_goals.append(goal)

            # Add now the variables that appear
            for var_name in used_variables:
                if var_name in profile:
                    yaml_goals.append({var_name: profile[var_name]})

            profile_yaml = {
                "test_name": profile["name"],
                "llm": {
                    "temperature": 0.8,
                    "model": "gpt-4o-mini",
                    "format": {"type": "text"},
                },
                "user": {
                    "language": primary_language,
                    "role": profile["role"],
                    "context": [
                        "personality: personalities/conversational-user.yml",
                    ],
                    "goals": yaml_goals,
                },
            }

            filename = (
                profile["name"]
                .lower()
                .replace(" ", "_")
                .replace(",", "")
                .replace("&", "and")
                + ".yaml"
            )
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as yf:
                yaml.dump(profile_yaml, yf, sort_keys=False, allow_unicode=True)

        print(f"Profiles saved in {output_dir}")

    else:
        print("No conversation goals were generated.")


if __name__ == "__main__":
    main()
