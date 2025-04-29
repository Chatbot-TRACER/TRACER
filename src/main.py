import os
import sys

from chatbot_connectors import ChatbotAdaUam, ChatbotTaskyto
from chatbot_explorer.analysis_orchestrator import (
    run_analysis_pipeline,
)
from chatbot_explorer.explorer import ChatbotExplorer
from utils.cli import parse_arguments
from utils.reporting import generate_graph_image, save_profiles, write_report


def main():
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
        print(f"Invalid technology: {technology}. Must be one of: {valid_technologies}")
        sys.exit(1)

    # Print configuration summary
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {technology}")
    print(f"Chatbot URL: {chatbot_url}")
    print(f"Exploration sessions: {max_sessions}")
    print(f"Max turns per session: {max_turns}")
    print(f"Using model: {model_name}")
    print(f"Output directory: {output_dir}")
    print("====================================")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the main explorer class
    explorer = ChatbotExplorer(model_name)

    # Store results from multiple sessions
    # Instantiate the correct chatbot connector based on technology
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)
    else:
        # This check is redundant due to earlier validation, but good practice
        print(f"Error: Unknown technology '{technology}'")
        sys.exit(1)

    # --- Run Full Exploration ---
    print("\n--- Starting Chatbot Exploration ---")
    exploration_results = explorer.run_full_exploration(
        chatbot_connector=the_chatbot,
        max_sessions=max_sessions,
        max_turns=max_turns,
    )
    print("--- Exploration Complete ---")

    # --- Run Analysis Pipeline ---
    print("\n--- Running Analysis Pipeline (Structure Inference & Profile Generation) ---")
    analysis_results = run_analysis_pipeline(
        explorer_instance=explorer, exploration_results=exploration_results
    )

    # Extract results for reporting and saving
    functionality_dicts = analysis_results["discovered_functionalities"]
    built_profiles = analysis_results["built_profiles"]

    # --- Save Generated Profiles using the new function ---
    save_profiles(built_profiles, output_dir)

    # --- Write Final Report ---
    write_report(
        output_dir,
        {"discovered_functionalities": functionality_dicts},
        exploration_results["supported_languages"],
        exploration_results["fallback_message"],
    )

    # --- Generate Workflow Graph Image ---
    graph_output_base = os.path.join(output_dir, "workflow_graph")
    generate_graph_image(functionality_dicts, graph_output_base)

    print("\n--- Exploration and Analysis Complete ---")


if __name__ == "__main__":
    main()
