"""Utilities for generating reports and visualizations from chatbot exploration."""

import html
import json
import os
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import graphviz
import yaml

from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()


@dataclass
class GraphBuildContext:
    graph: graphviz.Digraph
    processed_nodes: set[str] = field(default_factory=set)
    processed_edges: set[tuple[str, str]] = field(default_factory=set)
    node_clusters: dict[str, graphviz.Digraph] = field(default_factory=dict)


def export_graph(
    nodes: list[FunctionalityNode],
    output_path: str,
    fmt: str = "pdf",
    graph_font_size: int = 12,
    dpi: int = 300,
    compact: bool = False,
    top_down: bool = False,
):
    """Creates and renders a directed graph of chatbot functionality.

    Args:
        nodes: List of functionality nodes
        output_path: Path to save the graph
        fmt: Format to render (pdf, png, etc)
        graph_font_size: Font size for graph text elements
        dpi: Resolution of the output image in dots per inch
        compact: Whether to generate a more compact graph layout
        top_down: Whether to generate a top-down graph instead of left-to-right
    """
    if not nodes:
        return

    # Adjust DPI for SVG format to prevent oversized output
    adjusted_dpi = dpi
    if fmt.lower() == "svg":
        # For SVG, use 72 DPI since it's a vector format
        # High DPI values cause extremely large physical dimensions and scaling issues
        adjusted_dpi = 72  # Use 72 DPI for optimal SVG display

    dot = graphviz.Digraph(format=fmt)
    _set_graph_attributes(dot, graph_font_size, adjusted_dpi, compact, top_down)

    # Start node
    node_dim = 0.5
    dot.node(
        "start", label="", shape="circle", style="filled", fillcolor="black", width=str(node_dim), height=str(node_dim)
    )

    context = GraphBuildContext(graph=dot)

    # Group nodes by category
    nodes_by_category = {}
    for root in nodes:
        category = root.get("suggested_category", "Uncategorized")
        if category not in nodes_by_category:
            nodes_by_category[category] = []
        nodes_by_category[category].append(root)

    # Create a subgraph for each category
    for category, category_nodes in nodes_by_category.items():
        # Only create a cluster if we have more than one node in this category or total categories > 1
        if len(category_nodes) > 1 or len(nodes_by_category) > 1:
            cluster_name = f"cluster_{category.replace(' ', '_').lower()}"

            # Create a subgraph with a unique name for this category
            category_graph = graphviz.Digraph(name=cluster_name)
            category_graph.attr(
                label=category,
                style="rounded,filled",
                color="#DDDDDD",
                fillcolor="#F8F8F8:#EEEEEE",
                gradientangle="270",
                fontsize=str(graph_font_size + 1),
                fontname="Helvetica Neue, Helvetica, Arial, sans-serif",
                margin="15",
            )
            context.node_clusters[category] = category_graph

            # Process each node in this category
            for root_node in category_nodes:
                # Root nodes in a cluster are added to the category_graph
                # Their labels don't need to repeat the category, as the cluster shows it.
                _add_nodes(
                    ctx=context,
                    node=root_node,
                    parent="start",
                    depth=0,
                    graph_font_size=graph_font_size,
                    compact=compact,
                    target_graph=category_graph,
                    category_for_label=None,
                )

            # Add the subgraph to the main graph
            dot.subgraph(category_graph)
        else:
            # If only one node in category and it's the only category, don't create a cluster
            for root_node in category_nodes:
                # Root nodes not in a cluster are added to the main graph
                # Their labels should show their category.
                _add_nodes(
                    ctx=context,
                    node=root_node,
                    parent="start",
                    depth=0,
                    graph_font_size=graph_font_size,
                    compact=compact,
                    target_graph=context.graph,
                    category_for_label=root_node.get("suggested_category"),
                )

    try:
        # Suppress Graphviz warnings/errors to devnull because it clutters the terminal and things are getting properly rendered
        with open(os.devnull, "w", encoding="utf-8") as fnull:
            with redirect_stderr(fnull):
                dot.render(output_path, cleanup=True)
    except graphviz.backend.execute.ExecutableNotFound:
        raise RuntimeError(
            "Graphviz 'dot' executable not found. Ensure Graphviz is installed and in your system's PATH."
        )


def _set_graph_attributes(
    dot: graphviz.Digraph, graph_font_size: int = 12, dpi: int = 300, compact: bool = False, top_down: bool = False
) -> None:
    """Sets default attributes for the Graphviz graph, nodes, and edges.

    Args:
        dot: Graphviz Digraph object
        graph_font_size: Font size for graph text elements
        dpi: Resolution of the output image in dots per inch
        compact: Whether to generate a more compact graph layout
        top_down: Whether to generate a top-down graph instead of left-to-right
    """
    # Clean modern style
    bgcolor = "#ffffff"
    fontcolor = "#333333"
    edge_color = "#9DB2BF"
    font = "Helvetica Neue, Helvetica, Arial, sans-serif"

    # Calculate sizes relative to the font size
    node_fontsize = str(graph_font_size)
    graph_fontsize = str(graph_font_size + 1)  # Graph title slightly larger
    title_fontsize = str(graph_font_size + 2)  # Node titles larger than normal text

    # Adjust layout parameters based on compact mode and font size
    if graph_font_size >= 20:
        # Very tight spacing for large fonts
        pad = "0.3"
        nodesep = "0.3"
        ranksep = "0.5"
        splines = "ortho"
        overlap = "compress"
        node_margin = "0.1,0.08"  # Reduced margins for large fonts
    elif compact:
        # Compact mode - tighter spacing, orthogonal lines
        pad = "0.4"
        nodesep = "0.4"
        ranksep = "0.7"
        splines = "ortho"
        overlap = "compress"
        node_margin = "0.15,0.1"  # Slightly reduced margins for compact
    else:
        # Standard mode - more spacious layout, curved lines
        pad = "0.7"
        nodesep = "0.8"
        ranksep = "1.3"  # Slightly reduced from original 1.5
        splines = "curved"
        overlap = "false"
        node_margin = "0.2,0.15"  # Standard margins

    # Set graph orientation based on top_down parameter
    rankdir = "TB" if top_down else "LR"  # TB = Top to Bottom, LR = Left to Right

    dot.attr(
        rankdir=rankdir,
        bgcolor=bgcolor,
        fontname=font,
        fontsize=graph_fontsize,
        pad=pad,
        nodesep=nodesep,
        ranksep=ranksep,
        splines=splines,
        overlap=overlap,
        dpi=str(dpi),
        labelloc="t",
        fontcolor=fontcolor,
    )

    dot.attr(
        "node",
        shape="rectangle",
        style="filled,rounded",
        fontname=font,
        fontsize=node_fontsize,
        margin=node_margin,
        penwidth="1.5",
        fontcolor=fontcolor,
        height="0",  # Allow height to be determined by content
        width="0",  # Allow width to be determined by content
    )

    dot.attr(
        "edge",
        color=edge_color,
        penwidth="1.2",
        arrowsize="0.8",
        arrowhead="normal",
    )


def _get_node_style(depth: int) -> dict[str, str]:
    """Determines the node style based on its depth in the graph.

    Extended to support unlimited depth levels with a consistent color pattern.
    """
    color_schemes = {
        0: {"fillcolor": "#E3F2FD:#BBDEFB", "color": "#2196F3"},  # Blue theme
        1: {"fillcolor": "#E8F5E9:#C8E6C9", "color": "#4CAF50"},  # Green theme
        2: {"fillcolor": "#FFF8E1:#FFECB3", "color": "#FFC107"},  # Amber theme
        3: {"fillcolor": "#FFEBEE:#FFCDD2", "color": "#F44336"},  # Red theme
        4: {"fillcolor": "#F3E5F5:#E1BEE7", "color": "#9C27B0"},  # Purple theme
        5: {"fillcolor": "#E0F7FA:#B2EBF2", "color": "#00BCD4"},  # Cyan theme
        6: {"fillcolor": "#FFFDE7:#FFF9C4", "color": "#FFEB3B"},  # Yellow theme
        7: {"fillcolor": "#FBE9E7:#FFCCBC", "color": "#FF5722"},  # Deep Orange theme
        8: {"fillcolor": "#E8EAF6:#C5CAE9", "color": "#3F51B5"},  # Indigo theme
        9: {"fillcolor": "#F1F8E9:#DCEDC8", "color": "#8BC34A"},  # Light Green theme
    }

    # Depth modulo the number of color schemes to cycle through all available colors
    depth_mod = depth % len(color_schemes)
    return color_schemes[depth_mod]


def _add_nodes(
    ctx: GraphBuildContext,
    node: FunctionalityNode,
    parent: str,
    depth: int,
    graph_font_size: int = 12,
    compact: bool = False,
    target_graph: graphviz.Digraph = None,
    category_for_label: str | None = None,
):
    """Recursively adds nodes and edges to the graph."""
    name = node.get("name")
    if not name or name in ctx.processed_nodes:
        return

    # Build HTML label with outer brackets
    html_table = _build_label(node, graph_font_size, compact, category_to_display=category_for_label)
    label = f"<{html_table}>"

    target_graph.node(name, label=label, **_get_node_style(depth))
    ctx.processed_nodes.add(name)

    if (parent, name) not in ctx.processed_edges:
        if parent == "start" and target_graph != ctx.graph:
            ctx.graph.edge(parent, name)
        else:
            target_graph.edge(parent, name)
        ctx.processed_edges.add((parent, name))

    for child in node.get("children", []):
        # Children are always added to the main graph (ctx.graph).
        # Their labels should show their own category, if any.
        child_category_for_label = child.get("suggested_category")
        _add_nodes(
            ctx,
            child,
            parent=name,
            depth=depth + 1,
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=ctx.graph,  # Children go to the main graph
            category_for_label=child_category_for_label,
        )


def _truncate_text(text: str | None, max_length: int, already_escaped: bool = False) -> str:
    """Truncates text to a maximum length, adding ellipsis if truncated."""
    if text is None:
        return ""
    if len(text) > max_length:
        truncated = text[: max_length - 3].rstrip() + "..."
        return truncated if already_escaped else html.escape(truncated)
    return text if already_escaped else html.escape(text)


def _build_label(
    node: FunctionalityNode, graph_font_size: int = 12, compact: bool = False, category_to_display: str | None = None
) -> str:
    """Builds an HTML table with name, description, parameters, and outputs.

    Args:
        node: Functionality node
        graph_font_size: Font size for graph text elements
        compact: Whether to generate more compact node labels
        category_to_display: If provided, this category string will be displayed in the label.
    """
    title = html.escape(node.get("name", "").replace("_", " ").title())

    # Calculate font sizes based on the graph_font_size and compactness
    if compact:
        title_font_size = graph_font_size + 1  # Smaller title in compact mode
        normal_font_size = max(graph_font_size - 1, 8)  # Smaller content
        small_font_size = max(graph_font_size - 2, 7)  # Very small details
    else:
        title_font_size = graph_font_size + 2  # Larger title in standard mode
        normal_font_size = graph_font_size  # Normal content size
        small_font_size = max(graph_font_size - 1, 8)  # Smaller for details

    # Adjust truncation limits based on font size for better readability with large fonts
    if graph_font_size >= 20:
        # Very aggressive truncation for large fonts (20+)
        title_max_length = 25  # Increased from 15
        desc_max_length = 25
        output_combined_max_length = 30  # Combined name + description
        max_params = 2
        max_options = 2
        options_max_length = 15
    elif graph_font_size >= 16:
        # Moderate truncation for medium-large fonts (16-19)
        title_max_length = 30  # Increased from 20
        desc_max_length = 35
        output_combined_max_length = 40  # Combined name + description
        max_params = 3
        max_options = 3
        options_max_length = 20
    elif compact:
        # Original compact mode truncation for smaller fonts
        title_max_length = 40  # Increased from 30
        desc_max_length = 45
        output_combined_max_length = 50  # Combined name + description
        max_params = 3
        max_options = 3
        options_max_length = 25
    else:
        # Original standard mode truncation for smaller fonts
        title_max_length = 60  # Increased from 50
        desc_max_length = 70
        output_combined_max_length = 70  # Combined name + description
        max_params = 4
        max_options = 4
        options_max_length = 50

    # Truncate title if it's too long
    title = _truncate_text(title, title_max_length, already_escaped=True)

    rows = [f'<tr><td><font point-size="{title_font_size}"><b>{title}</b></font></td></tr>']

    # Add category if provided for display
    if category_to_display:
        rows.append(
            f'<tr><td><font color="#555555" point-size="{small_font_size}"><b>[{html.escape(category_to_display)}]</b></font></td></tr>'
        )

    # Add node description
    description = node.get("description")
    if description:
        truncated_desc = _truncate_text(description, desc_max_length)
        rows.append(
            f'<tr><td><font color="#777777" point-size="{small_font_size if compact else normal_font_size}"><i>{truncated_desc}</i></font></td></tr>'
        )

    # Process Parameters
    if compact:
        # Compact parameter display
        significant_params = []
        for p_data in node.get("parameters") or []:
            if isinstance(p_data, dict) and p_data.get("name"):
                significant_params.append(p_data)

        actual_param_rows = []
        shown_params = significant_params[:max_params]

        for p_data in shown_params:
            p_name = p_data.get("name", "")
            p_options = p_data.get("options", [])

            # Create the full line content first, then truncate the entire line consistently
            if isinstance(p_options, list) and len(p_options) > 0:
                options = [str(opt) for opt in p_options[:max_options]]
                options_str = ", ".join(options)
                if len(p_options) > max_options:
                    options_str += "..."
                full_line = f"{p_name}: {options_str}"

                # Apply consistent truncation based on font size to the entire line
                if graph_font_size >= 20:
                    line_max_length = 35  # Very aggressive for large fonts
                elif graph_font_size >= 16:
                    line_max_length = 45  # Moderate for medium-large fonts
                else:
                    line_max_length = 55  # Standard for smaller fonts

                if len(full_line) > line_max_length:
                    full_line = full_line[: line_max_length - 3] + "..."

                # Format with only parameter name in bold
                escaped_name = html.escape(p_name.replace("_", " "))
                escaped_options = html.escape(options_str)
                p_name_html = f"<b>{escaped_name}</b>: {escaped_options}"
            else:
                # Just parameter name - truncate if needed
                if graph_font_size >= 20:
                    line_max_length = 35
                elif graph_font_size >= 16:
                    line_max_length = 45
                else:
                    line_max_length = 55

                if len(p_name) > line_max_length:
                    p_name = p_name[: line_max_length - 3] + "..."

                p_name_html = f"<b>{html.escape(p_name.replace('_', ' '))}</b>"

            actual_param_rows.append(f'<tr><td><font point-size="{small_font_size}">{p_name_html}</font></td></tr>')

        # Show parameter count if there are more parameters than we're displaying
        if len(significant_params) > len(shown_params):
            more_count = len(significant_params) - len(shown_params)
            actual_param_rows.append(
                f'<tr><td><font point-size="{small_font_size}"><i>+{more_count} more params</i></font></td></tr>'
            )
    else:
        # Standard parameter display
        actual_param_rows = []
        for p_data in node.get("parameters") or []:
            if isinstance(p_data, dict):
                p_name = p_data.get("name")
                p_desc = p_data.get("description")
                p_options = p_data.get("options", [])

                # A parameter is significant if it has a name, or description, or non-empty options
                is_significant = bool(p_name or p_desc or (isinstance(p_options, list) and p_options))

                if is_significant:
                    # Apply consistent truncation based on font size to the entire line
                    if graph_font_size >= 20:
                        line_max_length = 35  # Very aggressive for large fonts
                    elif graph_font_size >= 16:
                        line_max_length = 45  # Moderate for medium-large fonts
                    else:
                        line_max_length = 70  # Standard for smaller fonts

                    # Create the formatted line with only name in bold
                    if isinstance(p_options, list) and p_options:
                        # Format: name: option1, option2, option3...
                        options_display = [str(opt) for opt in p_options[:max_options]]
                        options_str = ", ".join(options_display)
                        if len(p_options) > max_options:
                            options_str += "..."

                        full_line = f"{p_name}: {options_str}"
                        if len(full_line) > line_max_length:
                            # Calculate how much space we need for name + ": "
                            name_part = f"{p_name}: "
                            if len(name_part) < line_max_length - 3:
                                remaining_space = line_max_length - len(name_part) - 3
                                options_str = options_str[:remaining_space] + "..."
                            else:
                                # If name itself is too long, truncate the whole thing
                                full_line = full_line[: line_max_length - 3] + "..."
                                escaped_name = html.escape(p_name.replace("_", " ").title())
                                escaped_rest = html.escape(full_line[len(p_name) :])
                                p_line_html = f"<b>{escaped_name}</b>{escaped_rest}"

                        if "p_line_html" not in locals():
                            escaped_name = html.escape(p_name.replace("_", " ").title())
                            escaped_options = html.escape(options_str)
                            p_line_html = f"<b>{escaped_name}</b>: {escaped_options}"

                    elif p_desc:  # Has description
                        # Format: name: description
                        full_line = f"{p_name}: {p_desc}"
                        if len(full_line) > line_max_length:
                            # Calculate how much space we need for name + ": "
                            name_part = f"{p_name}: "
                            if len(name_part) < line_max_length - 3:
                                remaining_space = line_max_length - len(name_part) - 3
                                p_desc = p_desc[:remaining_space] + "..."
                            else:
                                # If name itself is too long, truncate the whole thing
                                full_line = full_line[: line_max_length - 3] + "..."
                                escaped_name = html.escape(p_name.replace("_", " ").title())
                                escaped_rest = html.escape(full_line[len(p_name) :])
                                p_line_html = f"<b>{escaped_name}</b>{escaped_rest}"

                        if "p_line_html" not in locals():
                            escaped_name = html.escape(p_name.replace("_", " ").title())
                            escaped_desc = html.escape(p_desc)
                            p_line_html = f"<b>{escaped_name}</b>: {escaped_desc}"

                    elif p_name:  # Has name, but no options and no description
                        # Just parameter name
                        if len(p_name) > line_max_length:
                            p_name = p_name[: line_max_length - 3] + "..."
                        p_line_html = f"<b>{html.escape(p_name.replace('_', ' ').title())}</b>"
                    else:
                        continue  # Skip if no meaningful content

                    actual_param_rows.append(
                        f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{p_line_html}</font></td></tr>'
                    )
                    # Reset for next iteration
                    if "p_line_html" in locals():
                        del p_line_html
            elif p_data is not None:  # Fallback for non-dict parameters
                actual_param_rows.append(
                    f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;<b>{html.escape(str(p_data))}</b></font></td></tr>'
                )

    if actual_param_rows:
        if not compact:
            rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Space before section
            rows.append("<HR/>")  # Horizontal rule
            rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Space after section
        rows.append(f'<tr><td><font point-size="{normal_font_size}"><u>Parameters</u></font></td></tr>')
        rows.extend(actual_param_rows)

    # Process Outputs
    actual_output_rows = []
    outputs_data = node.get("outputs") or []

    if outputs_data:
        # Process outputs exactly like parameters: name: description format
        for o_data in outputs_data:
            if isinstance(o_data, dict):
                o_category = o_data.get("category")
                o_desc = o_data.get("description")

                if o_category or o_desc:
                    # Format exactly like parameters: "category: description"
                    if o_category and o_desc:
                        full_line = f"{o_category}: {o_desc}"

                        # Truncate the entire line if it's too long (same logic as parameters)
                        if len(full_line) > output_combined_max_length:
                            # Calculate how much space we need for category + ": "
                            category_part = f"{o_category}: "
                            if len(category_part) < output_combined_max_length - 3:
                                remaining_space = output_combined_max_length - len(category_part) - 3
                                o_desc = o_desc[:remaining_space] + "..."
                            else:
                                # If category itself is too long, truncate the whole thing
                                full_line = full_line[: output_combined_max_length - 3] + "..."
                                escaped_category = html.escape(o_category.replace("_", " "))
                                escaped_rest = html.escape(full_line[len(o_category) :])
                                output_html = f"<b>{escaped_category}</b>{escaped_rest}"

                        if "output_html" not in locals():
                            escaped_category = html.escape(o_category.replace("_", " "))
                            escaped_desc = html.escape(o_desc)
                            output_html = f"<b>{escaped_category}</b>: {escaped_desc}"

                    elif o_category:
                        if len(o_category) > output_combined_max_length:
                            o_category = o_category[: output_combined_max_length - 3] + "..."
                        output_html = f"<b>{html.escape(o_category.replace('_', ' '))}</b>"
                    else:
                        if len(o_desc) > output_combined_max_length:
                            o_desc = o_desc[: output_combined_max_length - 3] + "..."
                        output_html = html.escape(o_desc)

                    # Format consistently with parameters
                    font_size = small_font_size if compact else normal_font_size
                    indent = "" if compact else "&nbsp;&nbsp;"

                    actual_output_rows.append(
                        f'<tr><td><font point-size="{font_size}">{indent}{output_html}</font></td></tr>'
                    )
                    # Reset for next iteration
                    if "output_html" in locals():
                        del output_html
            elif o_data is not None:
                # Non-dict outputs - just display as normal text, no bold
                full_line = str(o_data)
                if len(full_line) > output_combined_max_length:
                    full_line = full_line[: output_combined_max_length - 3] + "..."

                font_size = small_font_size if compact else normal_font_size
                indent = "" if compact else "&nbsp;&nbsp;"
                output_html = html.escape(full_line)

                actual_output_rows.append(
                    f'<tr><td><font point-size="{font_size}">{indent}{output_html}</font></td></tr>'
                )

    if actual_output_rows:
        # Limit outputs based on font size and compact mode for better space usage
        max_outputs_to_show = 3
        if graph_font_size >= 20:
            max_outputs_to_show = 2  # Very aggressive for large fonts
        elif graph_font_size >= 16:
            max_outputs_to_show = 2  # Moderate for medium-large fonts
        elif compact:
            max_outputs_to_show = 3  # Original compact logic

        if len(actual_output_rows) > max_outputs_to_show:
            shown_outputs = actual_output_rows[:max_outputs_to_show]
            remaining = len(actual_output_rows) - max_outputs_to_show
            shown_outputs.append(
                f'<tr><td><font point-size="{small_font_size}"><i>+{remaining} more outputs</i></font></td></tr>'
            )
            actual_output_rows = shown_outputs

        if not compact:
            # Add spacing and horizontal rule in standard mode
            rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')
            rows.append("<HR/>")
            rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')

        # Add heading and output rows
        output_title = "Outputs"
        if len(outputs_data) > max_outputs_to_show:
            output_title = f"Outputs ({len(outputs_data)})"
        rows.append(f'<tr><td><font point-size="{normal_font_size}"><u>{output_title}</u></font></td></tr>')
        rows.extend(actual_output_rows)

    # Return inner HTML for table (without extra brackets)
    return '<table border="0" cellborder="0" cellspacing="0">' + "".join(rows) + "</table>"


# ------------------------------------------------ #
# -------------------- REPORT -------------------- #
# ------------------------------------------------ #


def write_report(
    output_dir: str,
    structured_functionalities: list[FunctionalityNode],
    limitations: list[str],
    supported_languages: list[str],
    fallback_message: str | None,
    token_usage: dict[str, Any] = None,
) -> None:
    """Write analysis results to multiple report files.

    Args:
        output_dir: Directory to write the report files to
        structured_functionalities: List of functionalities representing the workflow structure
        limitations: List of discovered limitations
        supported_languages: List of detected supported languages
        fallback_message: Detected fallback message, or None if not found
        token_usage: Token usage statistics from LLM calls
    """
    output_path = Path(output_dir)

    # Write main report in Markdown format
    _write_main_report(output_path, structured_functionalities, supported_languages, fallback_message, token_usage)

    # Write raw JSON data to separate file
    _write_json_data(output_path, structured_functionalities)


# -------------------------------------------------- #
# -------------------- PROFILES -------------------- #
# -------------------------------------------------- #


def save_profiles(built_profiles: list[dict], output_dir: str) -> None:
    """Save user profiles as YAML files in the specified directory.

    Args:
        built_profiles: List of dictionaries representing user profiles
        output_dir: Directory to write the profile files to
    """
    if not built_profiles:
        logger.info("No user profiles to save")
        return

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_count = 0
    error_count = 0
    for profile in built_profiles:
        # Generate a safe filename from the test_name
        test_name = profile.get("test_name", f"profile_{hash(str(profile))}")  # Use hash as fallback

        if isinstance(test_name, dict):
            if test_name.get("function") == "random()" and "data" in test_name and test_name["data"]:
                # Use the first data element for a more descriptive random name
                base_name = str(test_name["data"][0])
                filename_base = f"random_profile_{base_name.lower().replace(' ', '_')}"
            else:
                # Fallback for other dict structures
                filename_base = f"profile_{hash(str(test_name))}"
        else:
            # Sanitize string test names for filenames
            filename_base = str(test_name).lower().replace(" ", "_").replace(",", "").replace("&", "and")

        # Basic sanitization to remove potentially problematic characters
        safe_filename_base = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in filename_base)
        filename = f"{safe_filename_base}.yaml"

        filepath = Path(output_dir) / filename
        try:
            with filepath.open("w", encoding="utf-8") as yf:
                yaml.dump(
                    profile,
                    yf,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=1000,
                )
            saved_count += 1
            logger.debug("Saved profile: %s", filename)
        except yaml.YAMLError:
            logger.exception("Failed to create YAML for profile '%s'.", test_name)
            error_count += 1
        except OSError:
            logger.exception("Failed to write file '%s'.", filename)
            error_count += 1

    if error_count:
        logger.warning("Saved %d profiles with %d errors", saved_count, error_count)
    else:
        logger.info("Successfully saved %d profiles to: %s/", saved_count, output_dir)


def _write_main_report(
    output_path: Path,
    structured_functionalities: list[FunctionalityNode],
    supported_languages: list[str],
    fallback_message: str | None,
    token_usage: dict[str, Any] = None,
) -> None:
    """Write the main analysis report in Markdown format."""
    report_path = output_path / "README.md"

    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# Chatbot Functionality Analysis\n\n")

            # Executive Summary
            _write_executive_summary(f, structured_functionalities, supported_languages)

            # Functionality Overview
            _write_functionality_overview(f, structured_functionalities)

            # Technical Details
            _write_technical_details(f, supported_languages, fallback_message)

            # Performance Statistics
            if token_usage:
                _write_performance_stats(f, token_usage)

            # Files Reference
            _write_files_reference(f)

        logger.info("Main report written to: %s", report_path)
    except OSError:
        logger.exception("Failed to write main report file.")


# Constants for category overview display
_MAX_FUNCTIONS_PER_CATEGORY = 5
_MAX_DESCRIPTION_LENGTH = 80


def _write_category_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write a category-based overview of main functions."""
    if not functionalities:
        f.write("No functionalities to overview.\n\n")
        return

    # Group all functions (including children) by category
    categories: dict[str, list[dict]] = {}

    def collect_by_category(nodes: list[FunctionalityNode]) -> None:
        for node in nodes:
            if isinstance(node, dict):
                category = node.get("suggested_category", "Uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(node)
                collect_by_category(node.get("children", []))

    collect_by_category(functionalities)

    # Sort categories and write overview
    sorted_categories = _get_sorted_categories(categories.keys())
    _write_category_sections(f, categories, sorted_categories)


def _get_sorted_categories(category_names: dict.keys) -> list[str]:
    """Sort categories alphabetically with Uncategorized last."""
    sorted_categories = sorted(category_names)
    if "Uncategorized" in sorted_categories:
        sorted_categories.remove("Uncategorized")
        sorted_categories.append("Uncategorized")
    return sorted_categories


def _write_category_sections(f: TextIO, categories: dict[str, list[dict]], sorted_categories: list[str]) -> None:
    """Write the sections for each category."""
    for category in sorted_categories:
        nodes = categories[category]
        icon = "📂" if category != "Uncategorized" else "📄"

        # Category header with count
        f.write(f"**{icon} {category}** ({len(nodes)} functions)\n")

        # Show representative functions
        _write_category_functions(f, nodes)
        f.write("\n")


def _write_category_functions(f: TextIO, nodes: list[dict]) -> None:
    """Write functions for a category with truncation if needed."""
    display_nodes = nodes[:_MAX_FUNCTIONS_PER_CATEGORY]
    for node in display_nodes:
        name = node.get("name", "Unnamed").replace("_", " ").title()
        desc = node.get("description", "No description")
        # Truncate long descriptions
        if len(desc) > _MAX_DESCRIPTION_LENGTH:
            desc = desc[: _MAX_DESCRIPTION_LENGTH - 3] + "..."
        f.write(f"- *{name}*: {desc}\n")

    # Show "and X more..." if there are more functions
    remaining_count = len(nodes) - _MAX_FUNCTIONS_PER_CATEGORY
    if remaining_count > 0:
        f.write(f"- *...and {remaining_count} more functions*\n")


def _write_executive_summary(f: TextIO, functionalities: list[FunctionalityNode], languages: list[str]) -> None:
    """Write executive summary section."""
    f.write("## 📊 TRACER Report\n\n")

    if not functionalities:
        f.write("❌ **No functionalities discovered**\n\n")
        return

    # Count functionalities
    total_functions = 0
    categories = set()

    def count_functions(nodes):
        nonlocal total_functions
        for node in nodes:
            if isinstance(node, dict):
                total_functions += 1
                if node.get("suggested_category"):
                    categories.add(node.get("suggested_category"))
                count_functions(node.get("children", []))

    count_functions(functionalities)

    f.write(f"✅ **{total_functions} functionalities** discovered across **{len(categories)} categories**\n\n")

    if languages:
        f.write(f"🌐 **Languages supported:** {', '.join(languages)}\n\n")

    # Category overview with key functions
    f.write("### 🎯 Functionality Overview\n\n")
    _write_category_overview(f, functionalities)


def _write_functionality_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write comprehensive functionality overview grouped by category with full details."""
    f.write("## 🗂️ Functionality Details\n\n")

    if not functionalities:
        f.write("No functionalities to categorize.\n\n")
        return

    # Group by category and collect all functions (including children)
    categories = {}

    def collect_by_category(nodes):
        for node in nodes:
            if isinstance(node, dict):
                category = node.get("suggested_category", "Uncategorized")
                if category not in categories:
                    categories[category] = []
                categories[category].append(node)
                collect_by_category(node.get("children", []))

    collect_by_category(functionalities)

    # Sort categories, put Uncategorized last
    sorted_categories = sorted(categories.keys())
    if "Uncategorized" in sorted_categories:
        sorted_categories.remove("Uncategorized")
        sorted_categories.append("Uncategorized")

    for category in sorted_categories:
        nodes = categories[category]
        icon = "📂" if category != "Uncategorized" else "📄"
        f.write(f"### {icon} {category} ({len(nodes)} functions)\n\n")

        for node in nodes:
            _write_detailed_function_info(f, node)
        f.write("\n")


def _write_detailed_function_info(f: TextIO, node: dict) -> None:
    """Write detailed information for a single function."""
    name = node.get("name", "Unnamed").replace("_", " ").title()
    desc = node.get("description", "No description")

    # Write function header
    f.write(f"#### 🔧 {name}\n\n")
    f.write(f"**Description:** {desc}\n\n")

    # Write parameters, outputs, and relationships
    _write_function_parameters(f, node)
    _write_function_outputs(f, node)
    _write_function_relationships(f, node)

    f.write("---\n\n")


def _write_function_parameters(f: TextIO, node: dict) -> None:
    """Write parameters section for a function."""
    parameters = node.get("parameters", [])
    if parameters and any(param for param in parameters if param is not None):
        f.write("**Parameters:**\n")
        for param in parameters:
            if param is not None:
                param_name = param.get("name", "Unknown")
                param_desc = param.get("description", "No description")
                param_options = param.get("options", [])

                f.write(f"- `{param_name}`: {param_desc}")
                if param_options:
                    options_str = ", ".join(f"`{opt}`" for opt in param_options)
                    f.write(f" *Options: {options_str}*")
                f.write("\n")
        f.write("\n")
    else:
        f.write("**Parameters:** None\n\n")


def _write_function_outputs(f: TextIO, node: dict) -> None:
    """Write outputs section for a function."""
    outputs = node.get("outputs", [])
    if outputs and any(output for output in outputs if output is not None):
        f.write("**Outputs:**\n")
        for output in outputs:
            if output is not None:
                output_category = output.get("category", "Unknown")
                output_desc = output.get("description", "No description")
                f.write(f"- `{output_category}`: {output_desc}\n")
        f.write("\n")
    else:
        f.write("**Outputs:** None\n\n")


def _write_function_relationships(f: TextIO, node: dict) -> None:
    """Write parent-child relationships for a function."""
    parent_names = node.get("parent_names", [])
    children = node.get("children", [])

    if parent_names:
        parents_str = ", ".join(f"`{parent.replace('_', ' ').title()}`" for parent in parent_names)
        f.write(f"**Parent Functions:** {parents_str}\n\n")

    if children:
        f.write("**Child Functions:**\n")
        for child in children:
            if isinstance(child, dict):
                child_name = child.get("name", "Unknown").replace("_", " ").title()
                child_desc = child.get("description", "No description")
                f.write(f"- `{child_name}`: {child_desc}\n")
        f.write("\n")


def _write_technical_details(f: TextIO, languages: list[str], fallback_message: str | None) -> None:
    """Write technical details section."""
    f.write("## ⚙️ Technical Details\n\n")

    # Language Support
    f.write("### 🌐 Language Support\n\n")
    if languages:
        for lang in languages:
            f.write(f"- {lang}\n")
    else:
        f.write("No specific language support detected.\n")
    f.write("\n")

    # Fallback Behavior
    f.write("### 🔄 Fallback Behavior\n\n")
    if fallback_message:
        f.write(f"```\n{fallback_message}\n```\n\n")
    else:
        f.write("No fallback message detected.\n\n")


def _write_performance_stats(f: TextIO, token_usage: dict[str, Any]) -> None:
    """Write performance statistics section."""
    f.write("## 📈 Performance Statistics\n\n")

    # Format numbers with commas
    def fmt_num(num):
        return f"{num:,}" if isinstance(num, (int, float)) else str(num)

    # Overview table
    f.write("### Overview\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Total LLM Calls | {fmt_num(token_usage.get('total_llm_calls', 'N/A'))} |\n")
    f.write(f"| Successful Calls | {fmt_num(token_usage.get('successful_llm_calls', 'N/A'))} |\n")
    f.write(f"| Failed Calls | {fmt_num(token_usage.get('failed_llm_calls', 'N/A'))} |\n")
    f.write(f"| Total Tokens | {fmt_num(token_usage.get('total_tokens_consumed', 'N/A'))} |\n")

    if "estimated_cost" in token_usage:
        f.write(f"| Estimated Cost | ${token_usage.get('estimated_cost', 0):.4f} USD |\n")

    if "total_application_execution_time" in token_usage:
        exec_time = token_usage["total_application_execution_time"]
        if isinstance(exec_time, dict) and "formatted" in exec_time:
            f.write(f"| Execution Time | {exec_time['formatted']} |\n")

    f.write("\n")

    # Phase breakdown
    f.write("### Phase Breakdown\n\n")

    phases = [
        ("Exploration", token_usage.get("exploration_phase", {})),
        ("Analysis", token_usage.get("analysis_phase", {})),
    ]

    f.write("| Phase | Prompt Tokens | Completion Tokens | Total Tokens | Cost |\n")
    f.write("|-------|---------------|-------------------|--------------|------|\n")

    for phase_name, phase_data in phases:
        prompt_tokens = fmt_num(phase_data.get("prompt_tokens", "N/A"))
        completion_tokens = fmt_num(phase_data.get("completion_tokens", "N/A"))
        total_tokens = fmt_num(phase_data.get("total_tokens", "N/A"))
        cost = f"${phase_data.get('estimated_cost', 0):.4f}" if "estimated_cost" in phase_data else "N/A"

        f.write(f"| {phase_name} | {prompt_tokens} | {completion_tokens} | {total_tokens} | {cost} |\n")

    f.write("\n")

    # Model information
    if token_usage.get("models_used"):
        f.write("### Models Used\n\n")
        for model in token_usage["models_used"]:
            f.write(f"- {model}\n")
        f.write("\n")


def _write_files_reference(f: TextIO) -> None:
    """Write files reference section."""
    f.write("## 📁 Generated Files\n\n")
    f.write("This analysis generated the following files:\n\n")
    f.write("- **`README.md`** - This main report with comprehensive functionality analysis\n")
    f.write("- **`functionalities.json`** - Raw JSON data structure\n")
    f.write("- **`workflow_graph.pdf`** - Visual graph of functionality relationships\n")
    f.write("- **`profiles/`** - Directory containing user profile YAML files\n\n")


def _write_json_data(output_path: Path, functionalities: list[FunctionalityNode]) -> None:
    """Write raw JSON data to separate file."""
    json_path = output_path / "functionalities.json"

    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(functionalities, f, indent=2, ensure_ascii=False)
        logger.info("JSON data written to: %s", json_path)
    except (TypeError, OSError) as e:
        logger.error("Failed to write JSON data: %s", e)
