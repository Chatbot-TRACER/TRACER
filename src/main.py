import sys
from pathlib import Path

from chatbot_explorer.agent import ChatbotExplorationAgent
from connectors.chatbot_connectors import ChatbotAdaUam, ChatbotTaskyto
from utils.cli import parse_arguments
from utils.reporting import generate_graph_image, save_profiles, write_report


def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Extract arguments for clarity
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_dir = args.output
    technology = args.technology

    # Validate technology choice
    if technology not in valid_technologies:
        print(f"Error: Invalid technology '{technology}'. Must be one of: {valid_technologies}")
        sys.exit(1)

    # Print configuration summary
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {technology}")
    print(f"Chatbot URL: {chatbot_url}")
    print(f"Exploration sessions: {max_sessions}")
    print(f"Max turns per session: {max_turns}")
    print(f"Using model: {model_name}")
    print("====================================")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Initialize the Agent ---
    print(f"\nInitializing Chatbot Exploration Agent with model: {model_name}...")
    try:
        agent = ChatbotExplorationAgent(model_name)
    except (ValueError, OSError) as e:
        print(f"Error initializing Chatbot Exploration Agent: {e}")
        print("Please ensure your API keys (e.g., OPENAI_API_KEY) are set correctly in your environment or .env file.")
        sys.exit(1)
    print("Agent initialized successfully.")

    # --- Instantiate the Chatbot Connector ---
    print(f"Instantiating connector for technology: {technology}")
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)
    else:
        # This case should not be reachable due to earlier validation
        print(f"Internal Error: Unknown technology '{technology}'")
        sys.exit(1)

    # --- Run Exploration ---
    print("\n--- Starting Chatbot Exploration Phase ---")
    try:
        exploration_results = agent.run_exploration(
            chatbot_connector=the_chatbot,
            max_sessions=max_sessions,
            max_turns=max_turns,
        )
        print("--- Exploration Phase Complete ---")
    except (ValueError, OSError, RuntimeError) as e:
        print("\n--- Error during Exploration Phase ---")
        print(f"Error: {e}")
        sys.exit(1)

    # --- Run Analysis ---
    print("\n--- Starting Analysis Phase (Structure Inference & Profile Generation) ---")
    try:
        # Pass the results from the exploration phase to the analysis method
        analysis_results = agent.run_analysis(exploration_results=exploration_results)
        print("--- Analysis Phase Complete ---")
    except (ValueError, KeyError) as e:
        print("\n--- Error during Analysis Phase ---")
        print(f"Error: {e}")
        sys.exit(1)

    # Extract results for reporting and saving
    built_profiles = analysis_results.get("built_profiles", [])

    # --- Save Generated Profiles ---
    print(f"\n--- Saving {len(built_profiles)} generated profiles ---")
    save_profiles(built_profiles, output_dir)

    # --- Write Final Report ---
    print("--- Writing final report ---")

    functionality_dicts = analysis_results.get("discovered_functionalities", {})
    limitations = analysis_results.get("discovered_limitations", [])
    supported_languages = exploration_results.get("supported_languages", ["N/A"])
    fallback_message = exploration_results.get("fallback_message", "N/A")

    write_report(
        output_dir,
        structured_functionalities=functionality_dicts,
        limitations=limitations,
        supported_languages=supported_languages,
        fallback_message=fallback_message,
    )

    # --- Generate Workflow Graph Image ---
    if functionality_dicts:
        print("--- Generating workflow graph image ---")
        graph_output_base = Path(output_dir) / "workflow_graph"
        generate_graph_image(functionality_dicts, str(graph_output_base))
    else:
        print("--- Skipping workflow graph image (no functionalities discovered) ---")

    print("\n--- Chatbot Explorer Finished ---")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
