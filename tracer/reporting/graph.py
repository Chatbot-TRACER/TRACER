"""Utilities for generating graph visualizations of chatbot functionality."""

import html
import os
import shutil
from contextlib import redirect_stderr
from pathlib import Path

import graphviz

from tracer.schemas.functionality_node_model import FunctionalityNode
from tracer.utils.logging_utils import get_logger
from tracer.utils.tracer_error import GraphvizNotInstalledError

from .config import (
    AddNodeOptions,
    FontSizeConfig,
    GraphBuildContext,
    GraphRenderOptions,
    GraphStyleConfig,
    TruncationConfig,
    create_font_size_config,
    create_layout_config,
    create_truncation_config,
)
from .constants import (
    COLOR_SCHEMES,
    DEFAULT_LINE_MAX_LENGTH,
    START_NODE_DIMENSION,
    SVG_DPI,
)

logger = get_logger()


def check_graphviz_availability() -> None:
    """Check if Graphviz is installed and accessible.

    This function performs an early check to ensure that Graphviz's 'dot'
    executable is available in the system's PATH. It uses a fail-early
    approach to detect missing Graphviz installations.

    Raises:
        GraphvizNotInstalledError: If Graphviz is not installed or not accessible.
    """
    if not shutil.which("dot"):
        logger.exception("Graphviz 'dot' executable not found.")
        msg = "Graphviz is not installed or not found in PATH. Please install Graphviz to enable graph generation."
        raise GraphvizNotInstalledError(msg)


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
    options: GraphRenderOptions | None = None,
) -> None:
    """Create and render a directed graph of chatbot functionality.

    This function generates a visual representation of chatbot functionality nodes
    as a directed graph using Graphviz. It supports various output formats and
    layout options.

    Args:
        nodes: List of functionality nodes to visualize
        output_path: Path where the graph should be saved (without extension)
        options: Graph rendering options. If None, uses default options.

    Raises:
        GraphvizNotInstalledError: If Graphviz is not installed or accessible
    """
    check_graphviz_availability()

    if options is None:
        options = GraphRenderOptions()
    _render_graph_with_options(nodes, output_path, options)


def _render_graph_with_options(
    nodes: list[FunctionalityNode],
    output_path: str,
    options: GraphRenderOptions,
) -> None:
    """Internal function to render graph with options dataclass."""
    if not nodes:
        logger.warning("No nodes provided for graph generation")
        return

    adjusted_dpi = adjust_dpi_for_format(options.fmt, options.dpi)

    dot = graphviz.Digraph(format=options.fmt)
    configure_graph_attributes(
        dot, options.graph_font_size, adjusted_dpi, compact=options.compact, top_down=options.top_down
    )

    create_start_node(dot)
    context = GraphBuildContext(graph=dot)
    nodes_by_category = group_nodes_by_category(nodes)
    process_categories(context, nodes_by_category, options.graph_font_size, compact=options.compact)

    render_graph(dot, output_path)


def configure_graph_attributes(
    dot: graphviz.Digraph, graph_font_size: int, dpi: int, *, compact: bool, top_down: bool
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
    layout_config = create_layout_config(graph_font_size, compact=compact)

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
    *,
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
            process_clustered_category(context, category, category_nodes, graph_font_size, compact=compact)
        else:
            process_unclustered_category(context, category_nodes, graph_font_size, compact=compact)


def process_clustered_category(
    context: GraphBuildContext,
    category: str,
    category_nodes: list[FunctionalityNode],
    graph_font_size: int,
    *,
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
        options = AddNodeOptions(
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=category_graph,
            category_for_label=None,
        )
        add_nodes(
            ctx=context,
            node=root_node,
            parent="start",
            depth=0,
            options=options,
        )

    context.graph.subgraph(category_graph)


def process_unclustered_category(
    context: GraphBuildContext, category_nodes: list[FunctionalityNode], graph_font_size: int, *, compact: bool
) -> None:
    """Process a category that should not be clustered.

    Args:
        context: Graph building context
        category_nodes: Nodes in this category
        graph_font_size: Font size for the graph
        compact: Whether to use compact layout
    """
    for root_node in category_nodes:
        options = AddNodeOptions(
            graph_font_size=graph_font_size,
            compact=compact,
            target_graph=context.graph,
            category_for_label=root_node.get("suggested_category"),
        )
        add_nodes(
            ctx=context,
            node=root_node,
            parent="start",
            depth=0,
            options=options,
        )


def render_graph(dot: graphviz.Digraph, output_path: str) -> None:
    """Render the graph to file.

    Args:
        dot: The configured Graphviz graph
        output_path: Output file path

    Raises:
        GraphvizNotInstalledError: If Graphviz executable is not found
    """
    try:
        # Suppress Graphviz warnings/errors to devnull
        with Path(os.devnull).open("w", encoding="utf-8") as fnull, redirect_stderr(fnull):
            dot.render(output_path, cleanup=True)

        # Log successful graph generation with format
        graph_format = dot.format or "pdf"
        final_output_path = f"{output_path}.{graph_format}"
        logger.info("Workflow graph saved to: %s", final_output_path)

    except graphviz.backend.execute.ExecutableNotFound as exc:
        logger.exception("Graphviz 'dot' executable not found during graph rendering")
        msg = "Graphviz 'dot' executable not found. Ensure Graphviz is installed and in your system's PATH."
        raise GraphvizNotInstalledError(msg) from exc


def add_nodes(
    ctx: GraphBuildContext,
    node: FunctionalityNode,
    parent: str,
    depth: int,
    options: AddNodeOptions,
) -> None:
    """Recursively add nodes and edges to the graph.

    Args:
        ctx: Graph building context
        node: Functionality node to add
        parent: Name of parent node
        depth: Depth level in graph
        options: Options for node addition
    """
    name = node.get("name")
    if not name or name in ctx.processed_nodes:
        return

    # Use the target_graph if provided, otherwise use the main graph
    target_graph = options.target_graph if options.target_graph is not None else ctx.graph

    # Build HTML label
    html_table = build_label(
        node,
        graph_font_size=options.graph_font_size,
        compact=options.compact,
        category_to_display=options.category_for_label,
    )
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
        child_options = AddNodeOptions(
            graph_font_size=options.graph_font_size,
            compact=options.compact,
            target_graph=ctx.graph,
            category_for_label=child.get("suggested_category"),
        )
        add_nodes(
            ctx,
            child,
            parent=name,
            depth=depth + 1,
            options=child_options,
        )


def truncate_text(text: str | None, max_length: int, *, already_escaped: bool = False) -> str:
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
    *,
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
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig, *, compact: bool = False
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
    """Build standard parameters display."""
    actual_param_rows = []
    for p_data in node.get("parameters") or []:
        if isinstance(p_data, dict):
            param_html = format_parameter_standard(p_data, trunc_config)
            if param_html:
                actual_param_rows.append(
                    f'<tr><td><font point-size="{font_config.normal_font_size}">  {param_html}</font></td></tr>'
                )
        elif p_data is not None:
            actual_param_rows.append(
                f'<tr><td><font point-size="{font_config.normal_font_size}">  <b>{html.escape(str(p_data))}</b></font></td></tr>'
            )
    if actual_param_rows:
        rows = [
            '<tr><td><font point-size="1"> </font></td></tr>',
            "<HR/>",
            '<tr><td><font point-size="1"> </font></td></tr>',
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
    node: FunctionalityNode, font_config: FontSizeConfig, trunc_config: TruncationConfig, *, compact: bool = False
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


def _format_category_only(o_category: str, trunc_config: TruncationConfig) -> str:
    """Format output with category only."""
    if len(o_category) > trunc_config.output_combined_max_length:
        o_category = o_category[: trunc_config.output_combined_max_length - 3] + "..."
    return f"<b>{html.escape(o_category.replace('_', ' '))}</b>"


def _format_description_only(o_desc: str, trunc_config: TruncationConfig) -> str:
    """Format output with description only."""
    if len(o_desc) > trunc_config.output_combined_max_length:
        o_desc = o_desc[: trunc_config.output_combined_max_length - 3] + "..."
    return html.escape(o_desc)


def _format_category_and_description(o_category: str, o_desc: str, trunc_config: TruncationConfig) -> str:
    """Format output with both category and description."""
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


def format_output(output_data: dict | str | None, trunc_config: TruncationConfig) -> str:
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
                return _format_category_and_description(o_category, o_desc, trunc_config)
            if o_category:
                return _format_category_only(o_category, trunc_config)
            return _format_description_only(o_desc, trunc_config)
    elif output_data is not None:
        # Non-dict outputs - just display as normal text, no bold
        full_line = str(output_data)
        if len(full_line) > trunc_config.output_combined_max_length:
            full_line = full_line[: trunc_config.output_combined_max_length - 3] + "..."
        return html.escape(full_line)

    return ""


def build_label(
    node: FunctionalityNode, *, graph_font_size: int = 12, compact: bool = False, category_to_display: str | None = None
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
    font_config = create_font_size_config(graph_font_size, compact=compact)
    trunc_config = create_truncation_config(graph_font_size, compact=compact)

    # Build title section
    rows = build_node_title(node, font_config, trunc_config, category_to_display, compact=compact)

    # Build parameters section
    param_rows = build_parameters_section(node, font_config, trunc_config, compact=compact)
    rows.extend(param_rows)

    # Build outputs section
    output_rows = build_outputs_section(node, font_config, trunc_config, compact=compact)
    rows.extend(output_rows)

    # Return inner HTML for table (without extra brackets)
    return '<table border="0" cellborder="0" cellspacing="0">' + "".join(rows) + "</table>"
