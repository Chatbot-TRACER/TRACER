import sys
import yaml
import os
import re
import json
import graphviz
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam

from typing import List, Dict, Any, Set

from utils.reporting import write_report, generate_graph_image


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
    if args.technology not in valid_technologies:
        print(
            f"Invalid technology: {args.technology}. Must be one of: {valid_technologies}"
        )
        sys.exit(1)

    # Print configuration summary
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {args.technology}")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output}")
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
        print(f"Error: Unknown technology '{args.technology}'")
        sys.exit(1)

    # --- Run Full Exploration ---
    print("\n--- Starting Chatbot Exploration ---")
    exploration_results = explorer.run_full_exploration(
        chatbot_connector=the_chatbot,
        max_sessions=args.sessions,
        max_turns=args.turns,
    )
    print("--- Exploration Complete ---")

    # Prepare state for LangGraph analysis (structure inference and profile generation)
    print(
        "\n--- Preparing to infer complete workflow structure and generate user profiles ---"
    )
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            },
        ],
        "conversation_history": exploration_results["conversation_sessions"],
        "discovered_functionalities": exploration_results[
            "root_nodes_dict"
        ],  # Use results from exploration
        "discovered_limitations": [],  # Limitations are not currently extracted during exploration
        "current_session": max_sessions,  # Keep max_sessions from args
        "exploration_finished": True,
        "conversation_goals": [],  # Not used in this flow currently
        "supported_languages": exploration_results["supported_languages"],
        "fallback_message": exploration_results["fallback_message"],
    }

    # 1. Infer the workflow structure using the dedicated graph
    print("\n--- Running workflow structure inference ---")
    structure_graph = explorer._build_structure_graph()
    structure_result = structure_graph.invoke(
        analysis_state, config={"configurable": {"thread_id": "structure_analysis"}}
    )

    # Use the refined structure from the structure graph for profile generation
    analysis_state["discovered_functionalities"] = structure_result[
        "discovered_functionalities"
    ]

    # Store the workflow structure for usage in profile generation
    workflow_structure = structure_result["discovered_functionalities"]

    # 2. Generate user profiles based on the inferred structure
    print("\n--- Generating user profiles ---")
    profile_graph = explorer._build_profile_generation_graph()

    # Add workflow structure to the state
    analysis_state["workflow_structure"] = workflow_structure

    result = profile_graph.invoke(
        analysis_state, config={"configurable": {"thread_id": "analysis_session"}}
    )

    # Update functionality_dicts with the final, structured version for reporting
    functionality_dicts = structure_result.get("discovered_functionalities", [])

    # --- Save Generated Profiles ---
    built_profiles = result.get("built_profiles", [])
    if built_profiles:
        print(f"\n--- Saving {len(built_profiles)} user profiles to disk ---")

        profiles_dir = output_dir  # Save profiles in the main output directory
        os.makedirs(profiles_dir, exist_ok=True)

        for profile in built_profiles:
            # Generate a safe filename from the test_name
            test_name = profile["test_name"]

            if isinstance(test_name, dict):
                # Handle structured test names (e.g., from random generation)
                if (
                    test_name.get("function") == "random()"
                    and "data" in test_name
                    and test_name["data"]
                ):
                    # Use the first data element for a more descriptive random name
                    base_name = str(test_name["data"][0])
                    filename = (
                        f"random_profile_{base_name.lower().replace(' ', '_')}.yaml"
                    )
                else:
                    # Fallback for other dict structures
                    filename = f"profile_{hash(str(test_name))}.yaml"
            else:
                # Sanitize string test names for filenames
                filename = (
                    str(test_name)
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

    # --- Write Final Report ---
    print("\n--- Writing report to disk ---")
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
