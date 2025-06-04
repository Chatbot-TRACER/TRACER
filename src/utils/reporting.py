"""Utilities for generating reports and visualizations from chatbot exploration.

This module provides comprehensive functionality for:
- Creating visual graph representations of chatbot functionality
- Generating detailed markdown report
- Saving user profiles in YAML format
- Managing graph layout and styling configurations
"""

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

# Constants for report formatting
MAX_FUNCTIONS_PER_CATEGORY = 5
MAX_DESCRIPTION_LENGTH = 80

# Graph configuration and styling constants
LARGE_FONT_THRESHOLD = 20
MEDIUM_FONT_THRESHOLD = 16
SVG_DPI = 72
START_NODE_DIMENSION = 0.5
DEFAULT_LINE_MAX_LENGTH = 55

# Color schemes for different depth levels
COLOR_SCHEMES = {
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


@dataclass
class FontSizeConfig:
    """Configuration for font sizes in graph nodes.

    Attributes:
        title_font_size: Font size for node titles
        normal_font_size: Font size for normal text
        small_font_size: Font size for small text and metadata
    """

    title_font_size: int
    normal_font_size: int
    small_font_size: int


@dataclass
class TruncationConfig:
    """Configuration for text truncation in graph nodes.

    Attributes:
        title_max_length: Maximum length for node titles
        desc_max_length: Maximum length for descriptions
        param_max_length: Maximum length for parameter names
        output_combined_max_length: Maximum length for combined output text
        max_params: Maximum number of parameters to show
        max_outputs: Maximum number of outputs to show
        max_options: Maximum number of parameter options to show
    """

    title_max_length: int = 30
    desc_max_length: int = 50
    param_max_length: int = 25
    output_combined_max_length: int = 40
    max_params: int = 3
    max_outputs: int = 3
    max_options: int = 2


@dataclass
class GraphStyleConfig:
    """Configuration for graph visual styling.

    Attributes:
        bgcolor: Background color for the graph
        fontcolor: Text color for nodes and labels
        edge_color: Color for graph edges
        font_family: Font family for text elements
    """

    bgcolor: str = "#ffffff"
    fontcolor: str = "#333333"
    edge_color: str = "#9DB2BF"
    font_family: str = "Helvetica Neue, Helvetica, Arial, sans-serif"


@dataclass
class GraphLayoutConfig:
    """Configuration for graph layout parameters.

    Attributes:
        pad: Padding around the graph
        nodesep: Separation between nodes at the same rank
        ranksep: Separation between ranks
        splines: Type of spline for edges
        overlap: How to handle node overlaps
        node_margin: Margin inside nodes
    """

    pad: str = "0.7"
    nodesep: str = "0.8"
    ranksep: str = "1.3"
    splines: str = "curved"
    overlap: str = "false"
    node_margin: str = "0.2,0.15"


@dataclass
class GraphBuildContext:
    """Context for tracking graph building state.

    Attributes:
        graph: The main Graphviz Digraph object
        processed_nodes: Set of node names that have been processed
        processed_edges: Set of edge tuples that have been processed
        node_clusters: Dictionary mapping category names to their subgraphs
    """

    graph: graphviz.Digraph
    processed_nodes: set[str] = field(default_factory=set)
    processed_edges: set[tuple[str, str]] = field(default_factory=set)
    node_clusters: dict[str, graphviz.Digraph] = field(default_factory=dict)


def create_font_size_config(graph_font_size: int, compact: bool) -> FontSizeConfig:
    """Create font size configuration based on graph settings.

    Args:
        graph_font_size: Base font size for the graph
        compact: Whether to use compact layout

    Returns:
        FontSizeConfig: Configuration object with calculated font sizes
    """
    if compact:
        return FontSizeConfig(
            title_font_size=graph_font_size + 1,
            normal_font_size=max(graph_font_size - 1, 8),
            small_font_size=max(graph_font_size - 2, 7),
        )
    return FontSizeConfig(
        title_font_size=graph_font_size + 2,
        normal_font_size=graph_font_size,
        small_font_size=max(graph_font_size - 1, 8),
    )


def create_truncation_config(graph_font_size: int, compact: bool) -> TruncationConfig:
    """Create text truncation configuration based on font size and layout.

    Args:
        graph_font_size: Base font size for the graph
        compact: Whether to use compact layout

    Returns:
        TruncationConfig: Configuration object with truncation limits
    """
    if graph_font_size >= LARGE_FONT_THRESHOLD:
        return TruncationConfig(
            title_max_length=25,
            desc_max_length=25,
            output_combined_max_length=30,
            max_params=2,
            max_options=2,
            max_outputs=2,
        )
    if graph_font_size >= MEDIUM_FONT_THRESHOLD:
        return TruncationConfig(
            title_max_length=30,
            desc_max_length=35,
            output_combined_max_length=40,
            max_params=3,
            max_options=3,
            max_outputs=2,
        )
    if compact:
        return TruncationConfig(
            title_max_length=40,
            desc_max_length=45,
            output_combined_max_length=50,
            max_params=3,
            max_options=3,
            max_outputs=3,
        )
    return TruncationConfig(
        title_max_length=60,
        desc_max_length=70,
        output_combined_max_length=70,
        max_params=4,
        max_options=4,
        max_outputs=3,
    )


def create_layout_config(graph_font_size: int, compact: bool) -> GraphLayoutConfig:
    """Create graph layout configuration based on font size and compactness.

    Args:
        graph_font_size: Base font size for the graph
        compact: Whether to use compact layout

    Returns:
        GraphLayoutConfig: Configuration object with layout parameters
    """
    if graph_font_size >= LARGE_FONT_THRESHOLD:
        return GraphLayoutConfig(
            pad="0.3", nodesep="0.3", ranksep="0.5", splines="ortho", overlap="compress", node_margin="0.1,0.08"
        )
    if compact:
        return GraphLayoutConfig(
            pad="0.4", nodesep="0.4", ranksep="0.7", splines="ortho", overlap="compress", node_margin="0.15,0.1"
        )
    return GraphLayoutConfig()  # Use defaults


def adjust_dpi_for_format(fmt: str, dpi: int) -> int:
    """Adjust DPI for specific output formats.

    Args:
        fmt: Output format (svg, pdf, png, etc.)
        dpi: Original DPI value

    Returns:
        int: Adjusted DPI value
    """
    if fmt.lower() == "svg":
        return SVG_DPI
    return dpi


def group_nodes_by_category(nodes: list[FunctionalityNode]) -> dict[str, list[FunctionalityNode]]:
    """Group functionality nodes by their suggested category.

    Args:
        nodes: List of functionality nodes to group

    Returns:
        dict: Dictionary mapping category names to lists of nodes
    """
    nodes_by_category: dict[str, list[FunctionalityNode]] = {}
    for node in nodes:
        category = node.get("suggested_category", "Uncategorized")
        if category not in nodes_by_category:
            nodes_by_category[category] = []
        nodes_by_category[category].append(node)
    return nodes_by_category


def should_create_cluster(category_nodes: list[FunctionalityNode], total_categories: int) -> bool:
    """Determine if a cluster should be created for a category.

    Args:
        category_nodes: Nodes in the category
        total_categories: Total number of categories

    Returns:
        bool: True if a cluster should be created
    """
    return len(category_nodes) > 1 or total_categories > 1


def create_category_cluster(category: str, graph_font_size: int) -> graphviz.Digraph:
    """Create a graphviz subgraph cluster for a category.

    Args:
        category: Category name
        graph_font_size: Font size for the graph

    Returns:
        graphviz.Digraph: Configured category subgraph
    """
    cluster_name = f"cluster_{category.replace(' ', '_').lower()}"
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
    return category_graph


def get_node_style(depth: int) -> dict[str, str]:
    """Get node style based on depth in the graph.

    Args:
        depth: Depth level in the graph

    Returns:
        dict: Style attributes for the node
    """
    depth_mod = depth % len(COLOR_SCHEMES)
    return COLOR_SCHEMES[depth_mod]


def export_graph(
    nodes: list[FunctionalityNode],
    output_path: str,
    fmt: str = "pdf",
    graph_font_size: int = 12,
    dpi: int = 300,
    compact: bool = False,
    top_down: bool = False,
) -> None:
    """Create and render a directed graph of chatbot functionality.

    This function generates a visual representation of chatbot functionality nodes
    as a directed graph using Graphviz. It supports various output formats and
    layout options.

    Args:
        nodes: List of functionality nodes to visualize
        output_path: Path where the graph should be saved (without extension)
        fmt: Output format (pdf, png, svg, etc.)
        graph_font_size: Base font size for text elements
        dpi: Resolution in dots per inch for raster formats
        compact: Whether to use a more compact layout
        top_down: Whether to use top-down layout instead of left-right

    Raises:
        RuntimeError: If Graphviz is not installed or accessible
    """
    if not nodes:
        logger.warning("No nodes provided for graph generation")
        return

    # Adjust DPI for specific formats
    adjusted_dpi = adjust_dpi_for_format(fmt, dpi)

    # Create and configure the main graph
    dot = graphviz.Digraph(format=fmt)
    configure_graph_attributes(dot, graph_font_size, adjusted_dpi, compact, top_down)

    # Create start node
    create_start_node(dot)

    # Initialize graph building context
    context = GraphBuildContext(graph=dot)

    # Group nodes by category and create clusters
    nodes_by_category = group_nodes_by_category(nodes)
    process_categories(context, nodes_by_category, graph_font_size, compact)

    # Render the graph
    render_graph(dot, output_path)


def configure_graph_attributes(
    dot: graphviz.Digraph, graph_font_size: int, dpi: int, compact: bool, top_down: bool
) -> None:
    """Configure graph attributes for styling and layout.

    Args:
        dot: Graphviz graph object to configure
        graph_font_size: Base font size for text elements
        dpi: Resolution for output
        compact: Whether to use compact layout
        top_down: Whether to use top-down orientation
    """
    style_config = GraphStyleConfig()
    layout_config = create_layout_config(graph_font_size, compact)

    # Set graph orientation
    rankdir = "TB" if top_down else "LR"

    # Configure main graph attributes
    dot.attr(
        rankdir=rankdir,
        bgcolor=style_config.bgcolor,
        fontname=style_config.font_family,
        fontsize=str(graph_font_size + 1),
        pad=layout_config.pad,
        nodesep=layout_config.nodesep,
        ranksep=layout_config.ranksep,
        splines=layout_config.splines,
        overlap=layout_config.overlap,
        dpi=str(dpi),
        labelloc="t",
        fontcolor=style_config.fontcolor,
    )

    # Configure node defaults
    dot.attr(
        "node",
        shape="rectangle",
        style="filled,rounded",
        fontname=style_config.font_family,
        fontsize=str(graph_font_size),
        margin=layout_config.node_margin,
        penwidth="1.5",
        fontcolor=style_config.fontcolor,
        height="0",
        width="0",
    )

    # Configure edge defaults
    dot.attr(
        "edge",
        color=style_config.edge_color,
        penwidth="1.2",
        arrowsize="0.8",
        arrowhead="normal",
    )


def create_start_node(dot: graphviz.Digraph) -> None:
    """Create the start node for the graph.

    Args:
        dot: The main Graphviz graph object
    """
    dot.node(
        "start",
        label="",
        shape="circle",
        style="filled",
        fillcolor="black",
        width=str(START_NODE_DIMENSION),
        height=str(START_NODE_DIMENSION),
    )


def process_categories(
    context: GraphBuildContext,
    nodes_by_category: dict[str, list[FunctionalityNode]],
    graph_font_size: int,
    compact: bool,
) -> None:
    """Process all categories and add them to the graph.

    Args:
        context: Graph building context
        nodes_by_category: Dictionary mapping categories to their nodes
        graph_font_size: Font size for the graph
        compact: Whether to use compact layout
    """
    for category, category_nodes in nodes_by_category.items():
        if should_create_cluster(category_nodes, len(nodes_by_category)):
            process_clustered_category(context, category, category_nodes, graph_font_size, compact)
        else:
            process_unclustered_category(context, category_nodes, graph_font_size, compact)


def process_clustered_category(
    context: GraphBuildContext,
    category: str,
    category_nodes: list[FunctionalityNode],
    graph_font_size: int,
    compact: bool,
) -> None:
    """Process a category that should be clustered.

    Args:
        context: Graph building context
        category: Category name
        category_nodes: Nodes in this category
        graph_font_size: Font size for the graph
        compact: Whether to use compact layout
    """
    category_graph = create_category_cluster(category, graph_font_size)
    context.node_clusters[category] = category_graph

    for root_node in category_nodes:
        add_nodes(
            ctx=context,
            node=root_node,
            parent="start",
            depth=0,
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=category_graph,
            category_for_label=None,
        )

    context.graph.subgraph(category_graph)


def process_unclustered_category(
    context: GraphBuildContext, category_nodes: list[FunctionalityNode], graph_font_size: int, compact: bool
) -> None:
    """Process a category that should not be clustered.

    Args:
        context: Graph building context
        category_nodes: Nodes in this category
        graph_font_size: Font size for the graph
        compact: Whether to use compact layout
    """
    for root_node in category_nodes:
        add_nodes(
            ctx=context,
            node=root_node,
            parent="start",
            depth=0,
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=context.graph,
            category_for_label=root_node.get("suggested_category"),
        )


def render_graph(dot: graphviz.Digraph, output_path: str) -> None:
    """Render the graph to file.

    Args:
        dot: The configured Graphviz graph
        output_path: Output file path

    Raises:
        RuntimeError: If Graphviz executable is not found
    """
    try:
        # Suppress Graphviz warnings/errors to devnull
        with open(os.devnull, "w", encoding="utf-8") as fnull:
            with redirect_stderr(fnull):
                dot.render(output_path, cleanup=True)
    except graphviz.backend.execute.ExecutableNotFound as exc:
        error_msg = "Graphviz 'dot' executable not found. Ensure Graphviz is installed and in your system's PATH."
        raise RuntimeError(error_msg) from exc


def add_nodes(
    ctx: GraphBuildContext,
    node: FunctionalityNode,
    parent: str,
    depth: int,
    graph_font_size: int = 12,
    *,
    compact: bool = False,
    target_graph: graphviz.Digraph | None = None,
    category_for_label: str | None = None,
) -> None:
    """Recursively add nodes and edges to the graph.

    Args:
        ctx: Graph building context
        node: Functionality node to add
        parent: Name of parent node
        depth: Depth level in graph
        graph_font_size: Font size for graph elements
        compact: Whether to use compact layout
        target_graph: Graph to add node to
        category_for_label: Category to display in label
    """
    name = node.get("name")
    if not name or name in ctx.processed_nodes:
        return

    # Use the target_graph if provided, otherwise use the main graph
    if target_graph is None:
        target_graph = ctx.graph

    # Build HTML label
    html_table = build_label(node, graph_font_size, compact, category_to_display=category_for_label)
    label = f"<{html_table}>"

    target_graph.node(name, label=label, **get_node_style(depth))
    ctx.processed_nodes.add(name)

    if (parent, name) not in ctx.processed_edges:
        if parent == "start" and target_graph != ctx.graph:
            ctx.graph.edge(parent, name)
        else:
            target_graph.edge(parent, name)
        ctx.processed_edges.add((parent, name))

    for child in node.get("children", []):
        child_category_for_label = child.get("suggested_category")
        add_nodes(
            ctx,
            child,
            parent=name,
            depth=depth + 1,
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=ctx.graph,
            category_for_label=child_category_for_label,
        )


def truncate_text(text: str | None, max_length: int, already_escaped: bool = False) -> str:
    """Truncate text to a maximum length, adding ellipsis if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        already_escaped: Whether text is already HTML-escaped

    Returns:
        str: Truncated and escaped text
    """
    if text is None:
        return ""
    if len(text) > max_length:
        truncated = text[: max_length - 3].rstrip() + "..."
        return truncated if already_escaped else html.escape(truncated)
    return text if already_escaped else html.escape(text)


def build_node_title(
    node: FunctionalityNode,
    font_config: FontSizeConfig,
    trunc_config: TruncationConfig,
    category_to_display: str | None = None,
    compact: bool = False,
) -> list[str]:
    """Build the title section of a node label.

    Args:
        node: Functionality node
        font_config: Font size configuration
        trunc_config: Text truncation configuration
        category_to_display: Category to display in label
        compact: Whether to use compact layout

    Returns:
        list[str]: HTML table rows for the title section
    """
    title = html.escape(node.get("name", "").replace("_", " ").title())
    title = truncate_text(title, trunc_config.title_max_length, already_escaped=True)

    rows = [f'<tr><td><font point-size="{font_config.title_font_size}"><b>{title}</b></font></td></tr>']

    # Add category if provided for display
    if category_to_display:
        rows.append(
            f'<tr><td><font color="#555555" point-size="{font_config.small_font_size}"><b>[{html.escape(category_to_display)}]</b></font></td></tr>'
        )

    # Add node description
    description = node.get("description")
    if description:
        truncated_desc = truncate_text(description, trunc_config.desc_max_length)
        font_size = font_config.small_font_size if compact else font_config.normal_font_size
        rows.append(f'<tr><td><font color="#777777" point-size="{font_size}"><i>{truncated_desc}</i></font></td></tr>')

    return rows


def build_parameters_section(
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig, compact: bool = False
) -> list[str]:
    """Build the parameters section of a node label.

    Args:
        node: Functionality node
        font_config: Font size configuration
        trunc_config: Text truncation configuration
        compact: Whether to use compact layout

    Returns:
        list[str]: HTML table rows for the parameters section
    """
    if compact:
        return build_compact_parameters(node, font_config, trunc_config)
    return build_standard_parameters(node, font_config, trunc_config)


def build_compact_parameters(
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig
) -> list[str]:
    """Build compact parameters display.

    Args:
        node: Functionality node
        font_config: Font size configuration
        trunc_config: Text truncation configuration

    Returns:
        list[str]: HTML table rows for compact parameters
    """
    significant_params = [
        p_data for p_data in node.get("parameters", []) if isinstance(p_data, dict) and p_data.get("name")
    ]

    if not significant_params:
        return []

    actual_param_rows = []
    shown_params = significant_params[: trunc_config.max_params]

    for p_data in shown_params:
        p_name = p_data.get("name", "")
        p_options = p_data.get("options", [])

        param_html = format_parameter_compact(p_name, p_options, trunc_config)
        actual_param_rows.append(
            f'<tr><td><font point-size="{font_config.small_font_size}">{param_html}</font></td></tr>'
        )

    # Show parameter count if there are more parameters than displayed
    if len(significant_params) > len(shown_params):
        more_count = len(significant_params) - len(shown_params)
        actual_param_rows.append(
            f'<tr><td><font point-size="{font_config.small_font_size}"><i>+{more_count} more params</i></font></td></tr>'
        )

    if actual_param_rows:
        rows = [f'<tr><td><font point-size="{font_config.normal_font_size}"><u>Parameters</u></font></td></tr>']
        rows.extend(actual_param_rows)
        return rows

    return []


def format_parameter_compact(param_name: str, param_options: list, trunc_config: TruncationConfig) -> str:
    """Format a parameter for compact display.

    Args:
        param_name: Name of the parameter
        param_options: List of parameter options
        trunc_config: Text truncation configuration

    Returns:
        str: HTML-formatted parameter string
    """
    if isinstance(param_options, list) and len(param_options) > 0:
        options = [str(opt) for opt in param_options[: trunc_config.max_options]]
        options_str = ", ".join(options)
        if len(param_options) > trunc_config.max_options:
            options_str += "..."
        full_line = f"{param_name}: {options_str}"

        # Apply consistent truncation based on predefined limits
        if len(full_line) > DEFAULT_LINE_MAX_LENGTH:
            full_line = full_line[: DEFAULT_LINE_MAX_LENGTH - 3] + "..."

        escaped_name = html.escape(param_name.replace("_", " "))
        escaped_options = html.escape(options_str)
        return f"<b>{escaped_name}</b>: {escaped_options}"
    # Just parameter name - truncate if needed
    if len(param_name) > DEFAULT_LINE_MAX_LENGTH:
        param_name = param_name[: DEFAULT_LINE_MAX_LENGTH - 3] + "..."
    return f"<b>{html.escape(param_name.replace('_', ' '))}</b>"


def build_standard_parameters(
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig
) -> list[str]:
    """Build standard parameters display.

    Args:
        node: Functionality node
        font_config: Font size configuration
        trunc_config: Text truncation configuration

    Returns:
        list[str]: HTML table rows for standard parameters
    """
    actual_param_rows = []

    for p_data in node.get("parameters") or []:
        if isinstance(p_data, dict):
            param_html = format_parameter_standard(p_data, trunc_config)
            if param_html:
                actual_param_rows.append(
                    f'<tr><td><font point-size="{font_config.normal_font_size}">&nbsp;&nbsp;{param_html}</font></td></tr>'
                )
        elif p_data is not None:
            # Fallback for non-dict parameters
            actual_param_rows.append(
                f'<tr><td><font point-size="{font_config.normal_font_size}">&nbsp;&nbsp;<b>{html.escape(str(p_data))}</b></font></td></tr>'
            )

    if actual_param_rows:
        rows = [
            '<tr><td><font point-size="1">&nbsp;</font></td></tr>',  # Space before section
            "<HR/>",  # Horizontal rule
            '<tr><td><font point-size="1">&nbsp;</font></td></tr>',  # Space after section
            f'<tr><td><font point-size="{font_config.normal_font_size}"><u>Parameters</u></font></td></tr>',
        ]
        rows.extend(actual_param_rows)
        return rows

    return []


def format_parameter_standard(param_data: dict, trunc_config: TruncationConfig) -> str:
    """Format a parameter for standard display.

    Args:
        param_data: Parameter data dictionary
        trunc_config: Text truncation configuration

    Returns:
        str: HTML-formatted parameter string, or empty if not significant
    """
    p_name = param_data.get("name")
    p_desc = param_data.get("description")
    p_options = param_data.get("options", [])

    # A parameter is significant if it has a name, description, or non-empty options
    is_significant = bool(p_name or p_desc or (isinstance(p_options, list) and p_options))

    if not is_significant:
        return ""

    if isinstance(p_options, list) and p_options:
        return format_parameter_with_options(p_name, p_options, trunc_config)
    if p_desc:
        return format_parameter_with_description(p_name, p_desc)
    if p_name:
        # Just parameter name
        if len(p_name) > DEFAULT_LINE_MAX_LENGTH:
            p_name = p_name[: DEFAULT_LINE_MAX_LENGTH - 3] + "..."
        return f"<b>{html.escape(p_name.replace('_', ' ').title())}</b>"

    return ""


def format_parameter_with_options(param_name: str, param_options: list, trunc_config: TruncationConfig) -> str:
    """Format parameter with options for standard display.

    Args:
        param_name: Parameter name
        param_options: List of parameter options
        trunc_config: Text truncation configuration

    Returns:
        str: HTML-formatted parameter with options
    """
    options_display = [str(opt) for opt in param_options[: trunc_config.max_options]]
    options_str = ", ".join(options_display)
    if len(param_options) > trunc_config.max_options:
        options_str += "..."

    full_line = f"{param_name}: {options_str}"
    if len(full_line) > DEFAULT_LINE_MAX_LENGTH:
        name_part = f"{param_name}: "
        if len(name_part) < DEFAULT_LINE_MAX_LENGTH - 3:
            remaining_space = DEFAULT_LINE_MAX_LENGTH - len(name_part) - 3
            options_str = options_str[:remaining_space] + "..."
        else:
            # If name itself is too long, truncate the whole thing
            full_line = full_line[: DEFAULT_LINE_MAX_LENGTH - 3] + "..."
            escaped_name = html.escape(param_name.replace("_", " ").title())
            escaped_rest = html.escape(full_line[len(param_name) :])
            return f"<b>{escaped_name}</b>{escaped_rest}"

    escaped_name = html.escape(param_name.replace("_", " ").title())
    escaped_options = html.escape(options_str)
    return f"<b>{escaped_name}</b>: {escaped_options}"


def format_parameter_with_description(param_name: str, param_desc: str) -> str:
    """Format parameter with description for standard display.

    Args:
        param_name: Parameter name
        param_desc: Parameter description

    Returns:
        str: HTML-formatted parameter with description
    """
    full_line = f"{param_name}: {param_desc}"
    if len(full_line) > DEFAULT_LINE_MAX_LENGTH:
        name_part = f"{param_name}: "
        if len(name_part) < DEFAULT_LINE_MAX_LENGTH - 3:
            remaining_space = DEFAULT_LINE_MAX_LENGTH - len(name_part) - 3
            param_desc = param_desc[:remaining_space] + "..."
        else:
            # If name itself is too long, truncate the whole thing
            full_line = full_line[: DEFAULT_LINE_MAX_LENGTH - 3] + "..."
            escaped_name = html.escape(param_name.replace("_", " ").title())
            escaped_rest = html.escape(full_line[len(param_name) :])
            return f"<b>{escaped_name}</b>{escaped_rest}"

    escaped_name = html.escape(param_name.replace("_", " ").title())
    escaped_desc = html.escape(param_desc)
    return f"<b>{escaped_name}</b>: {escaped_desc}"


def build_outputs_section(
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig, compact: bool = False
) -> list[str]:
    """Build the outputs section of a node label.

    Args:
        node: Functionality node
        font_config: Font size configuration
        trunc_config: Text truncation configuration
        compact: Whether to use compact layout

    Returns:
        list[str]: HTML table rows for the outputs section
    """
    outputs_data = node.get("outputs") or []
    if not outputs_data:
        return []

    actual_output_rows = []

    for o_data in outputs_data:
        output_html = format_output(o_data, trunc_config)
        if output_html:
            font_size = font_config.small_font_size if compact else font_config.normal_font_size
            indent = "" if compact else "&nbsp;&nbsp;"
            actual_output_rows.append(f'<tr><td><font point-size="{font_size}">{indent}{output_html}</font></td></tr>')

    if actual_output_rows:
        # Limit outputs based on configuration
        max_outputs_to_show = trunc_config.max_outputs
        if len(actual_output_rows) > max_outputs_to_show:
            shown_outputs = actual_output_rows[:max_outputs_to_show]
            remaining = len(actual_output_rows) - max_outputs_to_show
            shown_outputs.append(
                f'<tr><td><font point-size="{font_config.small_font_size}"><i>+{remaining} more outputs</i></font></td></tr>'
            )
            actual_output_rows = shown_outputs

        if not compact:
            # Add spacing and horizontal rule in standard mode
            rows = [
                '<tr><td><font point-size="1">&nbsp;</font></td></tr>',
                "<HR/>",
                '<tr><td><font point-size="1">&nbsp;</font></td></tr>',
            ]
        else:
            rows = []

        # Add heading and output rows
        output_title = "Outputs"
        if len(outputs_data) > max_outputs_to_show:
            output_title = f"Outputs ({len(outputs_data)})"
        rows.append(f'<tr><td><font point-size="{font_config.normal_font_size}"><u>{output_title}</u></font></td></tr>')
        rows.extend(actual_output_rows)
        return rows

    return []


def format_output(output_data: Any, trunc_config: TruncationConfig) -> str:
    """Format an output item for display.

    Args:
        output_data: Output data (dict or other)
        trunc_config: Text truncation configuration

    Returns:
        str: HTML-formatted output string
    """
    if isinstance(output_data, dict):
        o_category = output_data.get("category")
        o_desc = output_data.get("description")

        if o_category or o_desc:
            if o_category and o_desc:
                full_line = f"{o_category}: {o_desc}"

                # Truncate the entire line if it's too long
                if len(full_line) > trunc_config.output_combined_max_length:
                    category_part = f"{o_category}: "
                    if len(category_part) < trunc_config.output_combined_max_length - 3:
                        remaining_space = trunc_config.output_combined_max_length - len(category_part) - 3
                        o_desc = o_desc[:remaining_space] + "..."
                    else:
                        # If category itself is too long, truncate the whole thing
                        full_line = full_line[: trunc_config.output_combined_max_length - 3] + "..."
                        escaped_category = html.escape(o_category.replace("_", " "))
                        escaped_rest = html.escape(full_line[len(o_category) :])
                        return f"<b>{escaped_category}</b>{escaped_rest}"

                escaped_category = html.escape(o_category.replace("_", " "))
                escaped_desc = html.escape(o_desc)
                return f"<b>{escaped_category}</b>: {escaped_desc}"

            if o_category:
                if len(o_category) > trunc_config.output_combined_max_length:
                    o_category = o_category[: trunc_config.output_combined_max_length - 3] + "..."
                return f"<b>{html.escape(o_category.replace('_', ' '))}</b>"
            if len(o_desc) > trunc_config.output_combined_max_length:
                o_desc = o_desc[: trunc_config.output_combined_max_length - 3] + "..."
            return html.escape(o_desc)
    elif output_data is not None:
        # Non-dict outputs - just display as normal text, no bold
        full_line = str(output_data)
        if len(full_line) > trunc_config.output_combined_max_length:
            full_line = full_line[: trunc_config.output_combined_max_length - 3] + "..."
        return html.escape(full_line)

    return ""


def build_label(
    node: FunctionalityNode, graph_font_size: int = 12, compact: bool = False, category_to_display: str | None = None
) -> str:
    """Build an HTML table with name, description, parameters, and outputs.

    Args:
        node: Functionality node
        graph_font_size: Font size for graph text elements
        compact: Whether to generate more compact node labels
        category_to_display: If provided, this category string will be displayed in the label

    Returns:
        str: HTML table string for the node label
    """
    font_config = create_font_size_config(graph_font_size, compact)
    trunc_config = create_truncation_config(graph_font_size, compact)

    # Build title section
    rows = build_node_title(node, font_config, trunc_config, category_to_display, compact)

    # Build parameters section
    param_rows = build_parameters_section(node, font_config, trunc_config, compact)
    rows.extend(param_rows)

    # Build outputs section
    output_rows = build_outputs_section(node, font_config, trunc_config, compact)
    rows.extend(output_rows)

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
    write_main_report(output_path, structured_functionalities, supported_languages, fallback_message, token_usage)

    # Write raw JSON data to separate file
    write_json_data(output_path, structured_functionalities)


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


def write_main_report(
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
            write_executive_summary(f, structured_functionalities, supported_languages)

            # Functionality Overview
            write_functionality_overview(f, structured_functionalities)

            # Technical Details
            write_technical_details(f, supported_languages, fallback_message)

            # Performance Statistics
            if token_usage:
                write_performance_stats(f, token_usage)

            # Files Reference
            write_files_reference(f)

        logger.info("Main report written to: %s", report_path)
    except OSError:
        logger.exception("Failed to write main report file.")


def write_category_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
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
    sorted_categories = get_sorted_categories(categories.keys())
    write_category_sections(f, categories, sorted_categories)


def get_sorted_categories(category_names) -> list[str]:
    """Sort categories alphabetically with Uncategorized last."""
    sorted_categories = sorted(category_names)
    if "Uncategorized" in sorted_categories:
        sorted_categories.remove("Uncategorized")
        sorted_categories.append("Uncategorized")
    return sorted_categories


def write_category_sections(f: TextIO, categories: dict[str, list[dict]], sorted_categories: list[str]) -> None:
    """Write the sections for each category."""
    for category in sorted_categories:
        nodes = categories[category]
        icon = "📂" if category != "Uncategorized" else "📄"

        # Category header with count
        f.write(f"**{icon} {category}** ({len(nodes)} functions)\n")

        # Show representative functions
        write_category_functions(f, nodes)
        f.write("\n")


def write_category_functions(f: TextIO, nodes: list[dict]) -> None:
    """Write functions for a category with truncation if needed."""
    display_nodes = nodes[:MAX_FUNCTIONS_PER_CATEGORY]
    for node in display_nodes:
        name = node.get("name", "Unnamed").replace("_", " ").title()
        desc = node.get("description", "No description")
        # Truncate long descriptions
        if len(desc) > MAX_DESCRIPTION_LENGTH:
            desc = desc[: MAX_DESCRIPTION_LENGTH - 3] + "..."
        f.write(f"- *{name}*: {desc}\n")

    # Show "and X more..." if there are more functions
    remaining_count = len(nodes) - MAX_FUNCTIONS_PER_CATEGORY
    if remaining_count > 0:
        f.write(f"- *...and {remaining_count} more functions*\n")


def write_executive_summary(f: TextIO, functionalities: list[FunctionalityNode], languages: list[str]) -> None:
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
    write_category_overview(f, functionalities)


def write_functionality_overview(f: TextIO, functionalities: list[FunctionalityNode]) -> None:
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
            write_detailed_function_info(f, node)
        f.write("\n")


def write_detailed_function_info(f: TextIO, node: dict) -> None:
    """Write detailed information for a single function."""
    name = node.get("name", "Unnamed").replace("_", " ").title()
    desc = node.get("description", "No description")

    # Write function header
    f.write(f"#### 🔧 {name}\n\n")
    f.write(f"**Description:** {desc}\n\n")

    # Write parameters, outputs, and relationships
    write_function_parameters(f, node)
    write_function_outputs(f, node)
    write_function_relationships(f, node)

    f.write("---\n\n")


def write_function_parameters(f: TextIO, node: dict) -> None:
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


def write_function_outputs(f: TextIO, node: dict) -> None:
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


def write_function_relationships(f: TextIO, node: dict) -> None:
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


def write_technical_details(f: TextIO, languages: list[str], fallback_message: str | None) -> None:
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


def write_performance_stats(f: TextIO, token_usage: dict[str, Any]) -> None:
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


def write_files_reference(f: TextIO) -> None:
    """Write files reference section."""
    f.write("## 📁 Generated Files\n\n")
    f.write("This analysis generated the following files:\n\n")
    f.write("- **`README.md`** - This main report with comprehensive functionality analysis\n")
    f.write("- **`functionalities.json`** - Raw JSON data structure\n")
    f.write("- **`workflow_graph.pdf`** - Visual graph of functionality relationships\n")
    f.write("- **`profiles/`** - Directory containing user profile YAML files\n\n")


def write_json_data(output_path: Path, functionalities: list[FunctionalityNode]) -> None:
    """Write raw JSON data to separate file."""
    json_path = output_path / "functionalities.json"

    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(functionalities, f, indent=2, ensure_ascii=False)
        logger.info("JSON data written to: %s", json_path)
    except (TypeError, OSError) as e:
        logger.error("Failed to write JSON data: %s", e)
