import sys
import yaml
import os
import re
import json
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam
from chatbot_explorer.session import run_exploration_session
from chatbot_explorer.nodes.conversation_parameters_node import (
    generate_conversation_parameters,
)


def write_report(output_dir, result, supported_languages, fallback_message):
    """Write discovered functionalities, limitations, languages, and fallback message to report.txt."""
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

        f.write("## FUNCTIONALITIES\n")
        for i, func in enumerate(result.get("discovered_functionalities", []), 1):
            f.write(f"{i}. {func}\n")

        f.write("\n## LIMITATIONS\n")
        if "discovered_limitations" in result and result["discovered_limitations"]:
            for i, limitation in enumerate(result["discovered_limitations"], 1):
                f.write(f"{i}. {limitation}\n")
        else:
            f.write("No limitations discovered.\n")

        f.write("\n## SUPPORTED LANGUAGES\n")
        if supported_languages:
            for i, lang in enumerate(supported_languages, 1):
                f.write(f"{i}. {lang}\n")
        else:
            f.write("No specific language support detected.\n")

        f.write("\n## FALLBACK MESSAGE\n")
        f.write(
            fallback_message if fallback_message else "No fallback message detected.\n"
        )


def main():
    # Parse command line arguments
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Use the parameters from args
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_dir = args.output
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
    print(f"Output directory: {args.output}")
    print("====================================")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize explorer
    explorer = ChatbotExplorer(model_name)

    # Track multiple conversation sessions
    conversation_sessions = []
    supported_languages = []
    discovered_functionalities = []
    fallback_message = None

    # Create the chatbot according to the technology
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)

    # Run exploration sessions, each session is a conversation
    for session_num in range(max_sessions):
        # Run the exploration session
        conversation_history, new_languages, new_topics, new_fallback = (
            run_exploration_session(
                session_num,
                max_sessions,
                max_turns,
                explorer,
                the_chatbot,
                supported_languages,
                discovered_functionalities,
            )
        )

        # Store session data
        conversation_sessions.append(conversation_history)

        # Update supported languages if detected
        if session_num == 0 and new_languages:
            supported_languages = new_languages
            fallback_message = new_fallback

        # Update discovered functionalities
        for topic in new_topics:
            if topic not in discovered_functionalities:
                discovered_functionalities.append(topic)
                print(f"New functionality discovered: {topic}")

        # After session ends, save the conversation history
        print(f"\nSession {session_num + 1} complete")
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
        "fallback_message": fallback_message,
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

    # Save functionalities as a text file in the output directory
    write_report(output_dir, result, supported_languages, fallback_message)

    # Now save profiles from the built_profiles in result
    built_profiles = result.get("built_profiles", [])
    if built_profiles:
        print("\n--- Saving built user profiles to disk ---")
        profiles_dir = os.path.join(output_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        for profile in built_profiles:
            filename = (
                profile["test_name"]
                .lower()
                .replace(" ", "_")
                .replace(",", "")
                .replace("&", "and")
                + ".yaml"
            )
            filepath = os.path.join(profiles_dir, filename)
            with open(filepath, "w", encoding="utf-8") as yf:
                yaml.dump(profile, yf, sort_keys=False, allow_unicode=True)

    print(f"\nAll results saved to directory: {output_dir}")


if __name__ == "__main__":
    main()
