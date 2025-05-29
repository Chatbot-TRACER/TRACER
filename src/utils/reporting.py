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

    dot = graphviz.Digraph(format=fmt)
    _set_graph_attributes(dot, graph_font_size, dpi, compact, top_down)

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


def print_structured_functionalities(f: TextIO, nodes: list[FunctionalityNode], indent: str = "") -> None:
    """Recursively print the structured functionalities to a file object.

    Args:
        f: File object to write to
        nodes: List of node dictionaries representing functionalities
        indent: String used for indentation
    """
    for i, node in enumerate(nodes):
        # Defensive check: Ensure node is actually a dictionary
        if not isinstance(node, dict):
            f.write(f"{indent}ERROR: Expected a dictionary at index {i}, but got {type(node)}\n")
            continue  # Skip this element

        param_str = ""
        params_data = node.get("parameters", [])
        if params_data and isinstance(params_data, list):
            param_details = []
            for param in params_data:
                if isinstance(param, dict):
                    p_name = param.get("name", "N/A")
                    p_desc = param.get("description", "N/A")

                    # Add options to output if they exist
                    p_options = param.get("options", [])
                    options_str = f" [options: {', '.join(p_options)}]" if p_options else ""

                    param_details.append(f"{p_name}: {p_desc}{options_str}")
                else:
                    param_details.append(f"InvalidParamFormat({type(param)})")
            param_str = f" | Params: [{'; '.join(param_details)}]"
        elif params_data:  # If not a list but exists
            param_str = f" | Params: InvalidFormat({type(params_data)})"

        # Process output options
        output_str = ""
        outputs_data = node.get("outputs", [])
        if outputs_data and isinstance(outputs_data, list):
            output_details = []
            for output in outputs_data:
                if isinstance(output, dict):
                    o_category = output.get("category", "N/A")
                    o_desc = output.get("description", "N/A")
                    output_details.append(f"{o_category}: {o_desc}")
                else:
                    output_details.append(f"InvalidOutputFormat({type(output)})")
            output_str = f" | Outputs: [{'; '.join(output_details)}]"
        elif outputs_data:
            output_str = f" | Outputs: InvalidFormat({type(outputs_data)})"

        # Get category if available
        category_str = ""
        category = node.get("suggested_category")
        if category:
            category_str = f" | Category: {category}"

        node_name = node.get("name", "Unnamed Node")
        node_desc = node.get("description", "No description")
        f.write(f"{indent}- {node_name}: {node_desc}{category_str}{param_str}{output_str}\n")

        children = node.get("children", [])
        if children and isinstance(children, list):
            # Recursive call - ensure children is also List[Dict] before calling
            if all(isinstance(child, dict) for child in children):
                print_structured_functionalities(f, children, indent + "  ")
            else:
                f.write(f"{indent}  ERROR: Children of '{node_name}' contains non-dictionary elements.\n")
        elif children:  # If not a list but exists
            f.write(f"{indent}  ERROR: Children of '{node_name}' is not a list ({type(children)}).\n")


def _write_section_header(f: TextIO, header: str) -> None:
    """Write a section header to the report."""
    f.write(f"\n## {header}\n")


def _write_functionalities_section(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write the functionalities section to the report file."""
    _write_section_header(f, "FUNCTIONALITIES (Workflow Structure)")

    if not isinstance(functionalities, list):
        f.write(f"Functionality structure not in expected list format.\nType: {type(functionalities)}\n")
        return

    if not functionalities:
        f.write("No functionalities structure discovered (empty list).\n")
        return

    # Check if we're working with FunctionalityNode objects or already serialized dicts
    if hasattr(functionalities[0], "to_dict"):
        # Convert FunctionalityNode objects to dictionaries for printing
        serialized_nodes = [node.to_dict() for node in functionalities]
        print_structured_functionalities(f, serialized_nodes)
    elif isinstance(functionalities[0], dict):
        # Already serialized dictionaries
        print_structured_functionalities(f, functionalities)
    else:
        f.write("Functionality structure is a list, but elements are not recognized objects.\n")
        f.write(f"First element type: {type(functionalities[0])}\n")
        return


def _write_functionality_categories_section(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
    """Write the functionalities grouped by category to the report file."""
    _write_section_header(f, "FUNCTIONALITIES (By Category)")

    if not isinstance(functionalities, list):
        f.write(f"Functionality structure not in expected list format.\nType: {type(functionalities)}\n")
        return

    if not functionalities:
        f.write("No functionalities structure discovered (empty list).\n")
        return

    # Group nodes by their suggested category
    categories = {}

    def add_node_to_categories(node):
        if isinstance(node, dict):
            category = node.get("suggested_category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(node)

            # Process children recursively
            for child in node.get("children", []):
                add_node_to_categories(child)

    # Populate categories dictionary
    for node in functionalities:
        add_node_to_categories(node)

    # Display nodes by category
    if not categories:
        f.write("No categorized functionalities found.\n")
        return

    # Sort categories alphabetically, but put "Uncategorized" at the end if it exists
    sorted_categories = sorted(categories.keys())
    if "Uncategorized" in sorted_categories:
        sorted_categories.remove("Uncategorized")
        sorted_categories.append("Uncategorized")

    for category in sorted_categories:
        nodes = categories[category]
        f.write(f"\n### CATEGORY: {category} ({len(nodes)} functions)\n")

        # List all node names in this category
        for node in nodes:
            node_name = node.get("name", "Unnamed Node")
            node_desc = node.get("description", "No description")
            f.write(f"- {node_name}: {node_desc}\n")


def _write_json_section(f: TextIO, data: list[FunctionalityNode]) -> None:
    """Write the raw JSON structure section to the report file."""
    _write_section_header(f, "FUNCTIONALITIES (Raw JSON Structure)")

    if not isinstance(data, list):
        f.write("Functionality structure not in list format, cannot dump as JSON array.\n")
        f.write(f"Raw data (repr):\n{data!r}\n")
        return

    try:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))
    except TypeError as json_e:
        f.write(f"Could not serialize functionalities to JSON: {json_e}\n")
        f.write(f"Data (repr): {data!r}\n")


def _write_limitations_section(f: TextIO, limitations: list[str]) -> None:
    """Write the limitations section to the report file."""
    _write_section_header(f, "LIMITATIONS")

    if not isinstance(limitations, list):
        f.write(f"Limitations data not in expected list format.\nType: {type(limitations)}\n")
        return

    if not limitations:
        f.write("No limitations discovered (empty list).\n")
        return

    if not isinstance(limitations[0], str):
        f.write(f"Limitations list elements are not strings.\nFirst element type: {type(limitations[0])}\n")
        return

    for i, limitation in enumerate(limitations, 1):
        f.write(f"{i}. {limitation}\n")


def _write_languages_section(f: TextIO, languages: list[str]) -> None:
    """Write the supported languages section to the report file."""
    _write_section_header(f, "SUPPORTED LANGUAGES")

    if not isinstance(languages, list):
        f.write(f"Supported languages data not in expected list format.\nType: {type(languages)}\n")
        return

    if not languages:
        f.write("No specific language support detected (empty list).\n")
        return

    if not isinstance(languages[0], str):
        f.write(f"Supported languages list elements are not strings.\nFirst element type: {type(languages[0])}\n")
        return

    for i, lang in enumerate(languages, 1):
        f.write(f"{i}. {lang}\n")


def _write_fallback_section(f: TextIO, fallback_message: str | None) -> None:
    """Write the fallback message section to the report file."""
    _write_section_header(f, "FALLBACK MESSAGE")

    if isinstance(fallback_message, str):
        f.write(fallback_message)
    elif fallback_message is None:
        f.write("No fallback message detected.")
    else:
        f.write(f"Fallback message data is not a string or None.\nType: {type(fallback_message)}\n")
        f.write(f"Raw data: {fallback_message!r}\n")


def _write_token_usage_section(f: TextIO, token_usage: dict[str, Any]) -> None:
    """Write the token usage section to the report file.

    Args:
        f: File object to write to
        token_usage: Dictionary containing token usage statistics
    """
    _write_section_header(f, "\nTOKEN USAGE STATISTICS")

    if not isinstance(token_usage, dict):
        f.write(f"Token usage data not in expected dictionary format.\nType: {type(token_usage)}\n")
        return

    # Function to format token numbers with commas
    def format_num(num):
        return f"{num:,}" if isinstance(num, (int, float)) else str(num)

    # Write exploration phase token statistics
    exploration_data = token_usage.get("exploration_phase", {})
    f.write("EXPLORATION PHASE\n")
    f.write(f"  Prompt tokens:       {format_num(exploration_data.get('prompt_tokens', 'N/A'))}\n")
    f.write(f"  Completion tokens:   {format_num(exploration_data.get('completion_tokens', 'N/A'))}\n")
    f.write(f"  Total tokens:        {format_num(exploration_data.get('total_tokens', 'N/A'))}\n")
    if "estimated_cost" in exploration_data:
        f.write(f"  Estimated cost:      ${exploration_data.get('estimated_cost', 0):.4f} USD\n")

    # Write analysis phase token statistics
    analysis_data = token_usage.get("analysis_phase", {})
    f.write("\nANALYSIS PHASE\n")
    f.write(f"  Prompt tokens:       {format_num(analysis_data.get('prompt_tokens', 'N/A'))}\n")
    f.write(f"  Completion tokens:   {format_num(analysis_data.get('completion_tokens', 'N/A'))}\n")
    f.write(f"  Total tokens:        {format_num(analysis_data.get('total_tokens', 'N/A'))}\n")
    if "estimated_cost" in analysis_data:
        f.write(f"  Estimated cost:      ${analysis_data.get('estimated_cost', 0):.4f} USD\n")

    # Write total token usage statistics
    f.write("\nTOTAL TOKEN CONSUMPTION\n")
    f.write(f"  Total LLM calls:     {format_num(token_usage.get('total_llm_calls', 'N/A'))}\n")
    f.write(f"  Successful calls:    {format_num(token_usage.get('successful_llm_calls', 'N/A'))}\n")
    f.write(f"  Failed calls:        {format_num(token_usage.get('failed_llm_calls', 'N/A'))}\n")
    f.write(f"  Prompt tokens:       {format_num(token_usage.get('total_prompt_tokens', 'N/A'))}\n")
    f.write(f"  Completion tokens:   {format_num(token_usage.get('total_completion_tokens', 'N/A'))}\n")
    f.write(f"  Total tokens:        {format_num(token_usage.get('total_tokens_consumed', 'N/A'))}\n")

    # Add cost estimate if available
    if "estimated_cost" in token_usage:
        f.write(f"  Estimated cost:      ${token_usage.get('estimated_cost', 0):.4f} USD\n")

    # Add model information if available
    if token_usage.get("models_used"):
        models_str = ", ".join(token_usage["models_used"])
        f.write(f"\nMODELS USED\n  {models_str}\n")

    # Add cost model information if available
    if "cost_details" in token_usage and "cost_model_used" in token_usage["cost_details"]:
        f.write(f"\nCOST MODEL USED (for pricing calculation)\n  {token_usage['cost_details']['cost_model_used']}\n")

    # Add total application execution time if available
    if (
        "total_application_execution_time" in token_usage
        and isinstance(token_usage["total_application_execution_time"], dict)
        and "formatted" in token_usage["total_application_execution_time"]
    ):
        f.write(
            f"\nTOTAL APPLICATION EXECUTION TIME\n  {token_usage['total_application_execution_time']['formatted']} (HH:MM:SS)\n"
        )


def write_report(
    output_dir: str,
    structured_functionalities: list[FunctionalityNode],
    limitations: list[str],
    supported_languages: list[str],
    fallback_message: str | None,
    token_usage: dict[str, Any] = None,
) -> None:
    """Write analysis results to a report file.

    Args:
        output_dir: Directory to write the report file to
        structured_functionalities: List of functionalities representing the workflow structure
        limitations: List of discovered limitations
        supported_languages: List of detected supported languages
        fallback_message: Detected fallback message, or None if not found
        token_usage: Token usage statistics from LLM calls
    """
    report_path = Path(output_dir) / "report.txt"

    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

            _write_functionalities_section(f, structured_functionalities)
            _write_functionality_categories_section(f, structured_functionalities)
            _write_json_section(f, structured_functionalities)
            # Removed limitations section because we eneded up not using it
            _write_languages_section(f, supported_languages)
            _write_fallback_section(f, fallback_message)

            # Add token usage section if available
            if token_usage:
                _write_token_usage_section(f, token_usage)

        logger.info("Report successfully written to: %s", report_path)
    except OSError:
        logger.exception("Failed to write report file.")


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
