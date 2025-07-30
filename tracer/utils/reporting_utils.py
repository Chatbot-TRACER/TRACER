"""Reporting utility functions for logging and formatting output."""

from argparse import Namespace
from typing import Any

from tracer.utils.logging_utils import get_logger

logger = get_logger()


def log_configuration_summary(args: Namespace) -> None:
    """Log the configuration summary.

    Args:
        args: Parsed command line arguments
    """
    profile_model = args.profile_model or args.model

    logger.verbose("\n=== Chatbot Explorer Configuration ===")
    logger.verbose("Chatbot Technology:\t%s", args.technology)

    # Show connector parameters
    if args.connector_params:
        logger.verbose("Connector Parameters:\t%s", args.connector_params)
    else:
        logger.verbose("Connector Parameters:\tNone (using defaults)")

    logger.verbose("Exploration sessions:\t%d", args.sessions)
    logger.verbose("Max turns per session:\t%d", args.turns)
    logger.verbose("Exploration model:\t%s", args.model)
    logger.verbose("Profile model:\t\t%s", profile_model)
    logger.verbose("Output directory:\t%s", args.output)
    logger.verbose("Graph font size:\t\t%d", args.graph_font_size)
    logger.verbose("Compact graph:\t\t%s", "Yes" if args.compact else "No")
    logger.verbose("Graph orientation:\t%s", "Top-Down" if args.top_down else "Left-Right")
    logger.verbose("Graph format:\t\t%s", args.graph_format)
    logger.verbose("Nested forward chains:\t%s", "Yes" if args.nested_forward else "No")
    logger.verbose("======================================\n")


def log_token_usage_summary(token_usage: dict[str, Any]) -> None:
    """Log the final token usage summary.

    Args:
        token_usage: Dictionary containing token usage statistics
    """
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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into HH:MM:SS string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration as HH:MM:SS string
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
