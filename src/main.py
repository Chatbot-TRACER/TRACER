"""Main program entry point for the Chatbot Explorer."""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from chatbot_explorer.agent import ChatbotExplorationAgent
from chatbot_explorer.utils.logging_utils import get_logger, setup_logging
from connectors.chatbot_connectors import Chatbot, ChatbotAdaUam, ChatbotTaskyto
from utils.cli import parse_arguments
from utils.reporting import generate_graph_image, save_profiles, write_report

logger = get_logger()


def _setup_configuration() -> Namespace:
    """Parses command line arguments, validates config, and creates output dir.

    Returns:
        Namespace: The parsed and validated command line arguments.

    Raises:
        SystemExit: If the specified technology is invalid.
    """
    # Set up basic logging with default verbosity first
    setup_logging(0)  # Default to INFO level

    args = parse_arguments()

    if args.verbose > 0:
        setup_logging(args.verbose)

    valid_technologies = ["taskyto", "ada-uam"]

    if args.technology not in valid_technologies:
        logger.error("Invalid technology '%s'. Must be one of: %s", args.technology, valid_technologies)
        sys.exit(1)

    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)
    return args


def _initialize_agent(model_name: str) -> ChatbotExplorationAgent:
    """Initializes the Chatbot Exploration Agent.

    Handles potential errors during initialization, such as invalid API keys or
    connection issues during model loading.

    Args:
        model_name (str): The name of the language model to use.

    Returns:
        ChatbotExplorationAgent: The initialized agent instance.

    Raises:
        SystemExit: If agent initialization fails.
    """
    logger.info("Initializing Chatbot Exploration Agent with model: %s...", model_name)
    try:
        agent = ChatbotExplorationAgent(model_name)
    except Exception:
        logger.exception("Fatal Error initializing Chatbot Exploration Agent:")
        logger.exception("Please ensure API keys are set correctly and the model name is valid.")
        sys.exit(1)
    else:
        logger.info("Agent initialized successfully.")
        return agent


def _instantiate_connector(technology: str, url: str) -> Chatbot:
    """Instantiates the appropriate chatbot connector based on the specified technology.

    Args:
        technology (str): The name of the chatbot technology platform.
        url (str): The URL of the chatbot endpoint.

    Returns:
        Chatbot: An instance of the appropriate connector class.

    Raises:
        SystemExit: If the technology name is unknown (should be caught earlier).
    """
    logger.info("Instantiating connector for technology: %s", technology)
    if technology == "taskyto":
        return ChatbotTaskyto(url)
    if technology == "ada-uam":
        return ChatbotAdaUam(url)
    logger.error("Internal Error: Attempted to instantiate unknown technology '%s'", technology)
    sys.exit(1)


def _run_exploration_phase(
    agent: ChatbotExplorationAgent, chatbot_connector: Chatbot, max_sessions: int, max_turns: int
) -> dict[str, Any]:
    """Runs the chatbot exploration phase using the agent.

    Args:
        agent (ChatbotExplorationAgent): The initialized agent.
        chatbot_connector (Chatbot): The instantiated chatbot connector.
        max_sessions (int): Maximum number of exploration sessions.
        max_turns (int): Maximum turns per exploration session.

    Returns:
        Dict[str, Any]: The results collected during the exploration phase.

    Raises:
        SystemExit: If a critical error occurs during exploration.
    """
    logger.info("\n--- Starting Chatbot Exploration Phase ---")
    try:
        results = agent.run_exploration(
            chatbot_connector=chatbot_connector,
            max_sessions=max_sessions,
            max_turns=max_turns,
        )
        logger.info("--- Exploration Phase Complete ---")
    except Exception:
        logger.exception("--- Fatal Error during Exploration Phase ---")
        sys.exit(1)
    else:
        return results


def _run_analysis_phase(agent: ChatbotExplorationAgent, exploration_results: dict[str, Any]) -> dict[str, Any]:
    """Runs the analysis phase using the agent and exploration results.

    Args:
        agent (ChatbotExplorationAgent): The initialized agent.
        exploration_results (Dict[str, Any]): The results from the exploration phase.

    Returns:
        Dict[str, Any]: The results generated during the analysis phase.

    Raises:
        SystemExit: If a critical error occurs during analysis.
    """
    logger.info("--- Starting Analysis Phase (Structure Inference & Profile Generation) ---")
    try:
        results = agent.run_analysis(exploration_results=exploration_results)
        logger.info("--- Analysis Phase Complete ---")
    except Exception:
        logger.exception("--- Fatal Error during Analysis Phase ---")
        sys.exit(1)
    else:
        return results


def _generate_reports(
    output_dir: str,
    exploration_results: dict[str, Any],
    analysis_results: dict[str, Any],
) -> None:
    """Saves generated profiles, writes the final report, and generates the workflow graph image.

    Args:
        output_dir (str): The directory to save output files.
        exploration_results (Dict[str, Any]): Results from the exploration phase.
        analysis_results (Dict[str, Any]): Results from the analysis phase.
    """
    built_profiles = analysis_results.get("built_profiles", [])
    functionality_dicts = analysis_results.get("discovered_functionalities", {})
    limitations = analysis_results.get("discovered_limitations", [])
    supported_languages = exploration_results.get("supported_languages", ["N/A"])
    fallback_message = exploration_results.get("fallback_message", "N/A")

    logger.info("--- Saving %d generated profiles ---", len(built_profiles))

    save_profiles(built_profiles, output_dir)

    logger.info("--- Writing final report ---")
    write_report(
        output_dir,
        structured_functionalities=functionality_dicts,
        limitations=limitations,
        supported_languages=supported_languages,
        fallback_message=fallback_message,
    )

    if functionality_dicts:
        logger.info("--- Generating workflow graph image ---")
        graph_output_base = Path(output_dir) / "workflow_graph"
        try:
            generate_graph_image(functionality_dicts, str(graph_output_base))
        except Exception:
            logger.exception("Failed to generate workflow graph image")
    else:
        logger.info("--- Skipping workflow graph image (no functionalities discovered) ---")


def main() -> None:
    """Coordinates the setup, execution, and reporting for the Chatbot Explorer."""
    # 1. Parse arguments, validate inputs, create output dir
    args = _setup_configuration()

    # 2. Log Configuration Summary
    logger.verbose("\n=== Chatbot Explorer Configuration ===")
    logger.verbose("Chatbot Technology:\t%s", args.technology)
    logger.verbose("Chatbot URL:\t\t%s", args.url)
    logger.verbose("Exploration sessions:\t%d", args.sessions)
    logger.verbose("Max turns per session:\t%d", args.turns)
    logger.verbose("Using model:\t\t%s", args.model)
    logger.verbose("Output directory:\t%s", args.output)
    logger.verbose("======================================\n")

    # 4. Initialization
    agent = _initialize_agent(args.model)
    the_chatbot = _instantiate_connector(args.technology, args.url)

    # 5. Run Exploration
    exploration_results = _run_exploration_phase(agent, the_chatbot, args.sessions, args.turns)

    # 6. Run Analysis
    analysis_results = _run_analysis_phase(agent, exploration_results)

    # 7. Generate Reports
    _generate_reports(args.output, exploration_results, analysis_results)

    # 8. Finish
    logger.info("--- Chatbot Explorer Finished ---")
    logger.info("Results saved in: %s", args.output)


if __name__ == "__main__":
    main()
