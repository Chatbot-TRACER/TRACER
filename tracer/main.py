"""Main program entry point for the Chatbot Explorer."""

import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import requests
from chatbot_connectors import Chatbot, ChatbotFactory
from chatbot_connectors.cli import (
    handle_list_connector_params,
    handle_list_connectors,
)

from tracer.agent import ChatbotExplorationAgent
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
from tracer.utils.connector_utils import instantiate_connector
from tracer.utils.logging_utils import get_logger, setup_logging
from tracer.utils.reporting_utils import (
    format_duration,
    log_configuration_summary,
    log_token_usage_summary,
)
from tracer.utils.tracer_error import (
    ConnectorError,
    GraphvizNotInstalledError,
    LLMError,
    TracerError,
)

logger = get_logger()


def _setup_configuration() -> Namespace:
    """Parse command line arguments, validate config, and create output dir.

    Returns:
        The parsed and validated command line arguments

    Raises:
        TracerError: If the specified technology is invalid
    """
    # Set up basic logging with default verbosity first
    setup_logging(0)  # Default to INFO level

    args = parse_arguments()

    if args.verbose > 0:
        setup_logging(args.verbose)

    # Handle list-connector-params option
    if args.list_connector_params:
        try:
            handle_list_connector_params(args.list_connector_params)
            sys.exit(0)
        except (ValueError, RuntimeError):
            logger.exception("Failed to list connector parameters")
            sys.exit(1)

    # Handle list-connectors option
    if args.list_connectors:
        try:
            handle_list_connectors()
            sys.exit(0)
        except RuntimeError:
            logger.exception("Failed to list connectors")
            sys.exit(1)

    valid_technologies = ChatbotFactory.get_available_types()

    if args.technology not in valid_technologies:
        logger.error("Invalid technology '%s'. Must be one of: %s", args.technology, valid_technologies)
        msg = "Invalid technology."
        raise TracerError(msg)

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
        TracerError: If agent initialization fails.
    """
    logger.info("Initializing Chatbot Exploration Agent with model: %s...", model_name)
    try:
        agent = ChatbotExplorationAgent(model_name)
    except ImportError as e:
        logger.exception("Missing dependency for the selected model.")
        if "gemini" in model_name.lower():
            logger.exception(
                "To use Gemini models, install the required packages:"
                "\npip install langchain-google-genai google-generativeai"
            )
        msg = "Missing dependency for selected model."
        raise TracerError(msg) from e
    else:
        logger.info("Agent initialized successfully.")
        return agent


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
        TracerError: If a critical error occurs during exploration.
        requests.RequestException: If a connection error occurs.
    """
    logger.info("\n------------------------------------------")
    logger.info("--- Starting Chatbot Exploration Phase ---")
    logger.info("------------------------------------------")

    results = agent.run_exploration(
        chatbot_connector=chatbot_connector,
        max_sessions=max_sessions,
        max_turns=max_turns,
    )

    # Log token usage for exploration phase
    logger.info("\n=== Token Usage in Exploration Phase ===")
    logger.info(str(agent.token_tracker))

    return results


def _run_analysis_phase(
    agent: ChatbotExplorationAgent,
    exploration_results: dict[str, Any],
    *,
    nested_forward: bool = False,
    profile_model: str | None = None,
) -> dict[str, Any]:
    """Runs the analysis phase (structure inference and profile generation).

    Args:
        agent: The ChatbotExplorationAgent instance to use for analysis.
        exploration_results: Results from the exploration phase.
        nested_forward: Whether to use nested forward() chaining in variable definitions.
        profile_model: Model to use for profile generation (defaults to exploration model).

    Returns:
        Analysis results containing discovered functionalities and built profiles.

    Raises:
        TracerError: If a critical error occurs during analysis.
    """
    logger.info("\n-----------------------------------")
    logger.info("---   Starting Analysis Phase   ---")
    logger.info("-----------------------------------")

    # Mark the beginning of analysis phase for token tracking
    agent.token_tracker.mark_analysis_phase()

    results = agent.run_analysis(
        exploration_results=exploration_results, nested_forward=nested_forward, profile_model=profile_model
    )

    # Log token usage for analysis phase only
    logger.info("\n=== Token Usage in Analysis Phase ===")
    logger.info(str(agent.token_tracker))

    return results


def _generate_reports(results: ExecutionResults, config: ReportConfig) -> None:
    """Save generated profiles, write the final report, and generate the workflow graph image.

    Args:
        results: Container with all execution results
        config: Configuration for report generation
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
        except Exception as e:
            logger.exception("Failed to generate workflow graph image")
            msg = "Failed to generate workflow graph image."
            raise TracerError(msg) from e
    else:
        logger.info("--- Skipping workflow graph image (no functionalities discovered) ---")


def _run_tracer() -> None:
    """Coordinates the setup, execution, and reporting for the Chatbot Explorer."""
    app_start_time = time.monotonic()

    # Setup and configuration
    args = _setup_configuration()
    log_configuration_summary(args)

    # Initialize components
    agent = _initialize_agent(args.model)

    the_chatbot = instantiate_connector(args.technology, args.connector_params)

    # Execute phases
    exploration_results = _run_exploration_phase(agent, the_chatbot, args.sessions, args.turns)

    # Use profile_model if specified, otherwise use the same model as exploration
    profile_model = args.profile_model or args.model
    analysis_results = _run_analysis_phase(
        agent, exploration_results, nested_forward=args.nested_forward, profile_model=profile_model
    )

    # Calculate execution time and prepare results
    app_end_time = time.monotonic()
    total_app_duration_seconds = app_end_time - app_start_time
    formatted_app_duration = format_duration(total_app_duration_seconds)

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
    log_token_usage_summary(token_usage)

    logger.info("\n---------------------------------")
    logger.info("--- Chatbot Explorer Finished ---")
    logger.info("---------------------------------")


def main() -> None:
    """Top-level entry point for the Tracer application."""
    try:
        _run_tracer()
        logger.info("Tracer execution successful.")
    except GraphvizNotInstalledError:
        logger.exception("Graphviz dependency error")
        sys.exit(1)
    except ConnectorError:
        logger.exception("Chatbot connector error")
        sys.exit(1)
    except LLMError:
        logger.exception("Large Language Model API error")
        sys.exit(1)
    except TracerError:
        logger.exception("Tracer execution failed")
        sys.exit(1)
    except requests.RequestException:
        logger.exception("A connection error occurred")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected critical error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
