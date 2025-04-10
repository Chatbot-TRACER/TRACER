import sys
import yaml
import os
import re
import json
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam
from chatbot_explorer.session import run_exploration_session
from chatbot_explorer.functionality_node import FunctionalityNode
from typing import List, Dict, Any


def print_structured_functionalities(f, nodes: List[Dict[str, Any]], indent: str = ""):
    """
    Helper function to recursively print the structured functionalities.
    """
    for node in nodes:
        param_str_list = [p.get("name", "?") for p in node.get("parameters", [])]
        param_str = f" (Params: {', '.join(param_str_list)})" if param_str_list else ""
        f.write(
            f"{indent}- {node.get('name', 'N/A')}: {node.get('description', 'N/A')}{param_str}\n"
        )
        if node.get("children"):
            print_structured_functionalities(f, node["children"], indent + "  ")


def write_report(output_dir, result, supported_languages, fallback_message):
    """
    Write discovered functionalities (structured), limitations, languages, and fallback message.
    """
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

        f.write("## FUNCTIONALITIES (Workflow Structure)\n")
        # result['discovered_functionalities'] is now List[Dict[str, Any]] representing the roots
        structured_functionalities = result.get("discovered_functionalities", [])

        if structured_functionalities:
            # Check if it looks like our structured dicts
            if isinstance(structured_functionalities, list) and (
                not structured_functionalities
                or isinstance(structured_functionalities[0], dict)
            ):
                print_structured_functionalities(f, structured_functionalities)
            else:
                f.write("Functionality structure not in expected dictionary format.\n")
                f.write(
                    f"Raw data:\n{json.dumps(structured_functionalities, indent=2)}\n"
                )  # Print raw data
        else:
            f.write("No functionalities structure discovered.\n")

        # Add a raw JSON dump for detailed inspection
        f.write("\n## FUNCTIONALITIES (Raw JSON Structure)\n")
        if structured_functionalities:
            f.write(
                json.dumps(structured_functionalities, indent=2, ensure_ascii=False)
            )
        else:
            f.write("N/A\n")

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

    # Display brief summary of what was discovered
    print("\n=== Analysis Complete ===")
    print(f"Found {len(result.get('discovered_functionalities', []))} functionalities")
    print(f"Found {len(result.get('discovered_limitations', []))} limitations")
    print(
        f"Supported languages: {', '.join(supported_languages) if supported_languages else 'None detected'}"
    )

    # Save profiles from the built_profiles in result
    built_profiles = result.get("built_profiles", [])
    if built_profiles:
        print(f"\n--- Saving {len(built_profiles)} user profiles to disk ---")

        # If output_dir is directly specified, use it as the profile directory
        profiles_dir = output_dir
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
            print(f"  Saved profile: {filename}")

    print(f"\nAll profiles saved to: {output_dir}")

    print("\n--- Writing report to disk ---")
    write_report(
        output_dir,
        result,
        supported_languages,
        fallback_message,
    )


if __name__ == "__main__":
    main()
