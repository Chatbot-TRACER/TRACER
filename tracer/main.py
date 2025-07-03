"""Main program entry point for the Chatbot Explorer."""

import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from tracer.agent import ChatbotExplorationAgent
from tracer.connectors.chatbot_connectors import Chatbot, ChatbotFactory
from tracer.reporting import (
    ExecutionResults,
    GraphRenderOptions,
    ReportConfig,
    ReportData,
    export_graph,
    save_profiles,
    write_report,
)
from tracer.utils.cli import parse_arguments
from tracer.utils.logging_utils import get_logger, setup_logging

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

    valid_technologies = ChatbotFactory.get_available_types()

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
    except ImportError:
        logger.exception("Missing dependency")
        if "gemini" in model_name.lower():
            logger.exception(
                "To use Gemini models, install the required packages:"
                "\npip install langchain-google-genai google-generativeai"
            )
        sys.exit(1)
    except Exception:
        logger.exception("Fatal Error initializing Chatbot Exploration Agent:")

        if model_name.lower().startswith("gemini"):
            logger.exception(
                "For Gemini models, ensure the GOOGLE_API_KEY environment variable is set."
                "\nGet an API key at https://makersuite.google.com/app/apikey"
            )
        else:
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
        SystemExit: If the technology name is unknown or instantiation fails.
    """
    logger.info("Instantiating connector for technology: %s", technology)

    try:
        # Use the factory to check if URL is required and create the chatbot
        if ChatbotFactory.requires_url(technology):
            logger.info("Creating chatbot '%s' with base URL: %s", technology, url)
            return ChatbotFactory.create_chatbot(technology, base_url=url)

        logger.info("Creating chatbot '%s' without URL (pre-configured)", technology)
        return ChatbotFactory.create_chatbot(technology)

    except ValueError:
        logger.exception("Failed to instantiate connector for technology '%s'", technology)
        available_types = ChatbotFactory.get_available_types()
        logger.exception("Available chatbot types: %s", ", ".join(available_types))
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error instantiating connector for technology '%s'", technology)
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
    logger.info("\n------------------------------------------")
    logger.info("--- Starting Chatbot Exploration Phase ---")
    logger.info("------------------------------------------")

    try:
        results = agent.run_exploration(
            chatbot_connector=chatbot_connector,
            max_sessions=max_sessions,
            max_turns=max_turns,
        )

        # Log token usage for exploration phase
        logger.info("\n=== Token Usage in Exploration Phase ===")
        logger.info(str(agent.token_tracker))

    except Exception:
        logger.exception("--- Fatal Error during Exploration Phase ---")
        sys.exit(1)
    else:
        return results


def _run_analysis_phase(
    agent: ChatbotExplorationAgent, exploration_results: dict[str, Any], *, nested_forward: bool = False
) -> dict[str, Any]:
    """Runs the analysis phase using the agent and exploration results.

    Args:
        agent (ChatbotExplorationAgent): The initialized agent.
        exploration_results (Dict[str, Any]): The results from the exploration phase.
        nested_forward (bool): Whether to use nested forward() chaining in variable definitions.

    Returns:
        Dict[str, Any]: The results generated during the analysis phase.

    Raises:
        SystemExit: If a critical error occurs during analysis.
    """
    logger.info("\n-----------------------------------")
    logger.info("---   Starting Analysis Phase   ---")
    logger.info("-----------------------------------")

    # Mark the beginning of analysis phase for token tracking
    agent.token_tracker.mark_analysis_phase()

    try:
        results = agent.run_analysis(exploration_results=exploration_results, nested_forward=nested_forward)

        # Log token usage for analysis phase only
        logger.info("\n=== Token Usage in Analysis Phase ===")
        logger.info(str(agent.token_tracker))

    except Exception:
        logger.exception("--- Fatal Error during Analysis Phase ---")
        sys.exit(1)
    else:
        return results


def _generate_reports(results: ExecutionResults, config: ReportConfig) -> None:
    """Saves generated profiles, writes the final report, and generates the workflow graph image.

    Args:
        results (ExecutionResults): Container with all execution results.
        config (ReportConfig): Configuration for report generation.
    """
    built_profiles = results.analysis_results.get("built_profiles", [])
    functionality_dicts = results.analysis_results.get("discovered_functionalities", {})
    supported_languages = results.exploration_results.get("supported_languages", ["N/A"])
    fallback_message = results.exploration_results.get("fallback_message", "N/A")

    logger.info("\n--------------------------------")
    logger.info("---   Final Report Summary   ---")
    logger.info("--------------------------------\n")

    save_profiles(built_profiles, config.output_dir)

    report_data = ReportData(
        structured_functionalities=functionality_dicts,
        supported_languages=supported_languages,
        fallback_message=fallback_message,
        token_usage=results.token_usage,
    )

    write_report(config.output_dir, report_data)

    if functionality_dicts:
        graph_output_base = Path(config.output_dir) / "workflow_graph"
        try:
            # Determine which formats to export
            formats = ["pdf", "png", "svg"] if config.graph_format == "all" else [config.graph_format]

            # Export graphs in the specified format(s)
            for fmt in formats:
                options = GraphRenderOptions(
                    fmt=fmt,
                    graph_font_size=config.graph_font_size,
                    dpi=300,
                    compact=config.compact,
                    top_down=config.top_down,
                )
                export_graph(functionality_dicts, str(graph_output_base), options)
        except Exception:
            logger.exception("Failed to generate workflow graph image")
    else:
        logger.info("--- Skipping workflow graph image (no functionalities discovered) ---")


def _log_configuration_summary(args: Namespace) -> None:
    """Logs the configuration summary."""
    logger.verbose("\n=== Chatbot Explorer Configuration ===")
    logger.verbose("Chatbot Technology:\t%s", args.technology)
    logger.verbose("Chatbot URL:\t\t%s", args.url)
    logger.verbose("Exploration sessions:\t%d", args.sessions)
    logger.verbose("Max turns per session:\t%d", args.turns)
    logger.verbose("Using model:\t\t%s", args.model)
    logger.verbose("Output directory:\t%s", args.output)
    logger.verbose("Graph font size:\t\t%d", args.graph_font_size)
    logger.verbose("Compact graph:\t\t%s", "Yes" if args.compact else "No")
    logger.verbose("Graph orientation:\t%s", "Top-Down" if args.top_down else "Left-Right")
    logger.verbose("Graph format:\t\t%s", args.graph_format)
    logger.verbose("Nested forward chains:\t%s", "Yes" if args.nested_forward else "No")
    logger.verbose("======================================\n")


def _log_token_usage_summary(token_usage: dict[str, Any]) -> None:
    """Logs the final token usage summary."""
    exploration_data = token_usage.get("exploration_phase", {})
    analysis_data = token_usage.get("analysis_phase", {})

    logger.info("\n=== Token Usage Summary ===")

    logger.info("Exploration Phase:")
    logger.info("  Prompt tokens:     %s", f"{exploration_data.get('prompt_tokens', 0):,}")
    logger.info("  Completion tokens: %s", f"{exploration_data.get('completion_tokens', 0):,}")
    logger.info("  Total tokens:      %s", f"{exploration_data.get('total_tokens', 0):,}")
    logger.info("  Estimated cost:    $%.4f USD", exploration_data.get("estimated_cost", 0))

    logger.info("\nAnalysis Phase:")
    logger.info("  Prompt tokens:     %s", f"{analysis_data.get('prompt_tokens', 0):,}")
    logger.info("  Completion tokens: %s", f"{analysis_data.get('completion_tokens', 0):,}")
    logger.info("  Total tokens:      %s", f"{analysis_data.get('total_tokens', 0):,}")
    logger.info("  Estimated cost:    $%.4f USD", analysis_data.get("estimated_cost", 0))

    logger.info("\nTotal Consumption:")
    logger.info("  Total LLM calls:   %d", token_usage["total_llm_calls"])
    logger.info("  Successful calls:  %d", token_usage["successful_llm_calls"])
    logger.info("  Failed calls:      %d", token_usage["failed_llm_calls"])
    logger.info("  Prompt tokens:     %s", f"{token_usage['total_prompt_tokens']:,}")
    logger.info("  Completion tokens: %s", f"{token_usage['total_completion_tokens']:,}")
    logger.info("  Total tokens:      %s", f"{token_usage['total_tokens_consumed']:,}")
    logger.info("  Estimated cost:    $%.4f USD", token_usage.get("estimated_cost", 0))

    if token_usage.get("models_used"):
        logger.info("\nModels used: %s", ", ".join(token_usage["models_used"]))

    if (
        "total_application_execution_time" in token_usage
        and isinstance(token_usage["total_application_execution_time"], dict)
        and "formatted" in token_usage["total_application_execution_time"]
    ):
        logger.info("Total execution time: %s (HH:MM:SS)", token_usage["total_application_execution_time"]["formatted"])


def _format_duration(seconds: float) -> str:
    """Formats a duration in seconds into HH:MM:SS string."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    """Coordinates the setup, execution, and reporting for the Chatbot Explorer."""
    app_start_time = time.monotonic()

    # Setup and configuration
    args = _setup_configuration()
    _log_configuration_summary(args)

    # Initialize components
    agent = _initialize_agent(args.model)
    the_chatbot = _instantiate_connector(args.technology, args.url)

    # Execute phases
    exploration_results = _run_exploration_phase(agent, the_chatbot, args.sessions, args.turns)
    analysis_results = _run_analysis_phase(agent, exploration_results, nested_forward=args.nested_forward)

    # Calculate execution time and prepare results
    app_end_time = time.monotonic()
    total_app_duration_seconds = app_end_time - app_start_time
    formatted_app_duration = _format_duration(total_app_duration_seconds)

    token_usage = agent.token_tracker.get_summary()
    token_usage["total_application_execution_time"] = {
        "seconds": total_app_duration_seconds,
        "formatted": formatted_app_duration,
    }

    # Generate reports
    results = ExecutionResults(exploration_results, analysis_results, token_usage)
    config = ReportConfig(
        output_dir=args.output,
        graph_font_size=args.graph_font_size,
        compact=args.compact,
        top_down=args.top_down,
        graph_format=args.graph_format,
    )
    _generate_reports(results, config)

    # Final logging and cleanup
    _log_token_usage_summary(token_usage)

    logger.info("\n---------------------------------")
    logger.info("--- Chatbot Explorer Finished ---")
    logger.info("---------------------------------")


if __name__ == "__main__":
    main()
