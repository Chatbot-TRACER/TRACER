"""Command Line Interface utilities for parsing arguments."""

import argparse
from argparse import Namespace

from chatbot_connectors.cli import parse_connector_params


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chatbot Explorer - Discover functionalities of another chatbot")

    default_sessions = 3
    default_turns = 8
    default_model = "gpt-4o-mini"
    default_output_dir = "output"
    default_technology = "taskyto"
    default_graph_font_size = 12

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (-v for details, -vv for debug)",
    )

    parser.add_argument(
        "-s",
        "--sessions",
        type=int,
        default=default_sessions,
        help=f"Number of exploration sessions (default: {default_sessions})",
    )

    parser.add_argument(
        "-n",
        "--turns",
        type=int,
        default=default_turns,
        help=f"Maximum turns per session (default: {default_turns})",
    )

    parser.add_argument(
        "-t",
        "--technology",
        type=str,
        default=default_technology,
        help=f"Chatbot technology to use (default: {default_technology})",
    )

    parser.add_argument(
        "-cp",
        "--connector-params",
        type=parse_connector_params,
        default=None,
        help="Connector parameters as JSON string or key=value pairs separated by commas. "
        'Examples: \'{"base_url": "http://localhost", "port": 8080}\' or '
        '"base_url=http://localhost,port=8080". Use --list-connector-params <technology> to see required parameters for each connector.',
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=default_model,
        help=f"Model to use for exploration (default: {default_model}). Can be OpenAI models like 'gpt-4o' or Gemini models like 'gemini-2.0-flash'. Recommended: Use a more powerful model for better exploration quality.",
    )

    parser.add_argument(
        "-pm",
        "--profile-model",
        type=str,
        default=None,
        help="Model to use for profile generation (default: same as exploration model). Can be OpenAI models like 'gpt-4o' or Gemini models like 'gemini-2.0-flash'. Recommended: Use a cheaper model for cost optimization.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_output_dir,
        help=f"Output directory for results and profiles (default: {default_output_dir})",
    )

    parser.add_argument(
        "-gfs",
        "--graph-font-size",
        type=int,
        default=default_graph_font_size,
        help=f"Font size for graph text elements (default: {default_graph_font_size})",
    )

    parser.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="Generate a more compact graph with simplified nodes and tighter spacing",
    )

    parser.add_argument(
        "-td",
        "--top-down",
        action="store_true",
        help="Generate a top-down graph (instead of left-to-right) to better fit in papers",
    )

    parser.add_argument(
        "--graph-format",
        type=str,
        choices=["pdf", "png", "svg", "all"],
        default="pdf",
        help="Export format for the graph (default: pdf). Use 'all' to export in all formats",
    )

    parser.add_argument(
        "-nf",
        "--nested-forward",
        action="store_true",
        help="Make forward() functions nested in a chain to create more exhaustive profiles",
    )

    parser.add_argument(
        "--list-connector-params",
        type=str,
        metavar="TECHNOLOGY",
        help="List the available parameters for a specific chatbot technology and exit",
    )

    parser.add_argument(
        "--list-connectors",
        action="store_true",
        help="List all available chatbot connector technologies and exit",
    )

    return parser.parse_args()
