"""Script to generate a graph from JSON data without running the full analysis process.

Usage:
    python -m src.scripts.graph_generator --input output/report.txt --output output/custom_graph --font-size 25 --top-down --compact

This script extracts the JSON data from a report file or takes a direct JSON file path,
then generates the graph with the specified parameters.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the root directory to PATH
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# Import internal modules
from tracer.utils.logging_utils import get_logger
from tracer.utils.reporting import export_graph as _export_graph

logger = get_logger()


def extract_json_from_report(report_path: str) -> list:
    """Extract the JSON data from a report file.

    The report file typically has sections with the JSON data under 'FUNCTIONALITIES (Raw JSON Structure)'.
    """
    json_section_marker = "## FUNCTIONALITIES (Raw JSON Structure)"
    next_section_marker = "##"

    with Path(report_path).open(encoding="utf-8") as f:
        content = f.read()

    # Extract the section between JSON marker and the next section
    if json_section_marker in content:
        json_start = content.find(json_section_marker) + len(json_section_marker)
        json_end = content.find(next_section_marker, json_start)

        # If no next section, take all remaining text
        if json_end == -1:
            json_end = len(content)

        json_str = content[json_start:json_end].strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON from report")
            raise
    else:
        error_msg = f"Could not find JSON section marker '{json_section_marker}' in the report file"
        raise ValueError(error_msg)


def load_json_data(path: str) -> list:
    """Load JSON data from a file, either a report or a direct JSON file."""
    if path.endswith(".json"):
        with Path(path).open(encoding="utf-8") as f:
            return json.load(f)
    else:
        # Assume it's a report file
        return extract_json_from_report(path)


def save_json_data(nodes: list, output_path: str) -> None:
    """Save the JSON data to a file."""
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
    logger.info("JSON data saved to %s", output_path)


def export_graph(nodes: list, **kwargs: str | int | bool) -> None:
    """Wrapper around the export_graph function from reporting.py.

    This function handles the case where nodes are plain dictionaries
    instead of FunctionalityNode objects.
    """
    return _export_graph(nodes, **kwargs)


def main() -> None:
    """Main function to parse arguments and generate the graph."""
    parser = argparse.ArgumentParser(description="Generate a graph from JSON data")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to the JSON file or report file containing the JSON data"
    )
    parser.add_argument("--output", "-o", required=True, help="Path to save the graph (without extension)")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf", help="Output format (default: pdf)")
    parser.add_argument("--font-size", type=int, default=14, help="Font size for the graph text (default: 14)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of the output image in DPI (default: 300)")
    parser.add_argument("--compact", action="store_true", help="Generate a more compact graph layout")
    parser.add_argument("--top-down", action="store_true", help="Generate a top-down graph instead of left-to-right")
    parser.add_argument("--save-json", help="Path to save the extracted JSON data (optional)")

    args = parser.parse_args()

    try:
        # Load the JSON data
        logger.info("Loading JSON data from %s", args.input)
        nodes = load_json_data(args.input)

        if not nodes:
            logger.error("No functionality nodes found in the input file")
            return

        # Save JSON data if requested
        if args.save_json:
            save_json_data(nodes, args.save_json)

        # Generate the graph
        logger.info("Generating graph with font size %d, DPI %d", args.font_size, args.dpi)
        logger.info("Output format: %s", args.format)
        logger.info("Compact mode: %s", args.compact)
        logger.info("Top-down layout: %s", args.top_down)

        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_dir = output_path.parent
        if output_dir != Path():
            output_dir.mkdir(parents=True, exist_ok=True)

        export_graph(
            nodes=nodes,
            output_path=args.output,
            fmt=args.format,
            graph_font_size=args.font_size,
            dpi=args.dpi,
            compact=args.compact,
            top_down=args.top_down,
        )

        logger.info("Graph generated successfully: %s.%s", args.output, args.format)

    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        logger.exception("Error generating graph")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error generating graph")
        sys.exit(1)


if __name__ == "__main__":
    main()
