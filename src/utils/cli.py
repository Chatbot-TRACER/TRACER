"""Command Line Interface utilities for parsing arguments."""

import argparse
from argparse import Namespace


def parse_arguments() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chatbot Explorer - Discover functionalities of another chatbot")

    default_sessions = 3
    default_turns = 8
    default_url = "http://localhost:5000"
    default_model = "gpt-4o-mini"
    default_output_dir = "output"
    default_technology = "taskyto"

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
        "-u",
        "--url",
        type=str,
        default=default_url,
        help=f"Chatbot URL to explore (default: {default_url})",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=default_model,
        help=f"OpenAI model to use (default: {default_model})",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_output_dir,
        help=f"Output directory for results and profiles (default: {default_output_dir})",
    )

    return parser.parse_args()
