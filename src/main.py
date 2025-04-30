import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from chatbot_explorer.agent import ChatbotExplorationAgent
from connectors.chatbot_connectors import Chatbot, ChatbotAdaUam, ChatbotTaskyto
from utils.cli import parse_arguments
from utils.reporting import generate_graph_image, save_profiles, write_report


def _setup_configuration() -> Namespace:
    """Parses arguments, validates config, prints summary, and creates output dir."""
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Validate technology choice
    if args.technology not in valid_technologies:
        print(f"Error: Invalid technology '{args.technology}'. Must be one of: {valid_technologies}")
        sys.exit(1)

    # Print configuration summary
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {args.technology}")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print("====================================")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    return args


def _initialize_agent(model_name: str) -> ChatbotExplorationAgent:
    """Initializes the Chatbot Exploration Agent with error handling."""
    print(f"\nInitializing Chatbot Exploration Agent with model: {model_name}...")
    try:
        agent = ChatbotExplorationAgent(model_name)
    except (ValueError, OSError) as e:
        print(f"Error initializing Chatbot Exploration Agent: {e}")
        print("Please ensure your API keys (e.g., OPENAI_API_KEY) are set correctly in your environment or .env file.")
        sys.exit(1)
    else:
        print("Agent initialized successfully.")
        return agent


def _instantiate_connector(technology: str, url: str) -> Chatbot:
    """Instantiates the appropriate chatbot connector."""
    print(f"Instantiating connector for technology: {technology}")
    if technology == "taskyto":
        return ChatbotTaskyto(url)
    if technology == "ada-uam":
        return ChatbotAdaUam(url)
    # This case should not be reachable due to earlier validation in _setup_configuration
    print(f"Internal Error: Unknown technology '{technology}'")
    sys.exit(1)


def _run_exploration_phase(
    agent: ChatbotExplorationAgent, chatbot_connector: Chatbot, max_sessions: int, max_turns: int
) -> dict[str, Any]:
    """Runs the exploration phase with error handling."""
    print("\n--- Starting Chatbot Exploration Phase ---")
    try:
        results = agent.run_exploration(
            chatbot_connector=chatbot_connector,
            max_sessions=max_sessions,
            max_turns=max_turns,
        )
        print("--- Exploration Phase Complete ---")
    except (ValueError, OSError, RuntimeError) as e:
        print("\n--- Error during Exploration Phase ---")
        print(f"Error: {e}")
        sys.exit(1)
    else:
        return results


def _run_analysis_phase(agent: ChatbotExplorationAgent, exploration_results: dict[str, Any]) -> dict[str, Any]:
    """Runs the analysis phase with error handling."""
    print("\n--- Starting Analysis Phase (Structure Inference & Profile Generation) ---")
    try:
        results = agent.run_analysis(exploration_results=exploration_results)
        print("--- Analysis Phase Complete ---")
    except (ValueError, KeyError) as e:
        print("\n--- Error during Analysis Phase ---")
        print(f"Error: {e}")
        sys.exit(1)
    else:
        return results


def _generate_reports(
    output_dir: str,
    exploration_results: dict[str, Any],
    analysis_results: dict[str, Any],
) -> None:
    """Saves profiles, writes the report, and generates the graph."""
    built_profiles = analysis_results.get("built_profiles", [])
    functionality_dicts = analysis_results.get("discovered_functionalities", {})
    limitations = analysis_results.get("discovered_limitations", [])
    supported_languages = exploration_results.get("supported_languages", ["N/A"])
    fallback_message = exploration_results.get("fallback_message", "N/A")

    # Save Generated Profiles
    print(f"\n--- Saving {len(built_profiles)} generated profiles ---")
    save_profiles(built_profiles, output_dir)

    # Write Final Report
    print("--- Writing final report ---")
    write_report(
        output_dir,
        structured_functionalities=functionality_dicts,
        limitations=limitations,
        supported_languages=supported_languages,
        fallback_message=fallback_message,
    )

    # Generate Workflow Graph Image
    if functionality_dicts:
        print("--- Generating workflow graph image ---")
        graph_output_base = Path(output_dir) / "workflow_graph"
        generate_graph_image(functionality_dicts, str(graph_output_base))
    else:
        print("--- Skipping workflow graph image (no functionalities discovered) ---")


def main() -> None:
    """Main execution function."""
    # 1. Parses arguments, validates config, prints summary, and creates output dir.
    args = _setup_configuration()

    # 2. Initialization
    agent = _initialize_agent(args.model)
    the_chatbot = _instantiate_connector(args.technology, args.url)

    # 3. Run Exploration
    exploration_results = _run_exploration_phase(agent, the_chatbot, args.sessions, args.turns)

    # 4. Run Analysis
    analysis_results = _run_analysis_phase(agent, exploration_results)

    # 5. Generate Report, Save Profiles, and Generate Graph
    _generate_reports(args.output, exploration_results, analysis_results)

    # 6. Finish
    print("\n--- Chatbot Explorer Finished ---")
    print(f"Results saved in: {args.output}")


if __name__ == "__main__":
    main()
