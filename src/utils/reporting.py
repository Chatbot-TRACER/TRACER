"""Utilities for generating reports and visualizations from chatbot exploration."""

import html
import json
import os
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import graphviz
import yaml

from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_DESCRIPTION_LENGTH = 70
MAX_OUTPUTS_LENGTH = 50
MAX_OPTIONS = 4
MAX_OPTIONS_STRING_LENGTH = 50


@dataclass
class GraphBuildContext:
    graph: graphviz.Digraph
    processed_nodes: set[str] = field(default_factory=set)
    processed_edges: set[tuple[str, str]] = field(default_factory=set)
    node_clusters: dict[str, graphviz.Digraph] = field(default_factory=dict)


def export_graph(nodes: list[FunctionalityNode], output_path: str, fmt: str = "pdf", graph_font_size: int = 12):
    """Creates and renders a directed graph of chatbot functionality.

    Args:
        nodes: List of functionality nodes
        output_path: Path to save the graph
        fmt: Format to render (pdf, png, etc)
        graph_font_size: Font size for graph text elements
    """
    if not nodes:
        return

    dot = graphviz.Digraph(format=fmt)
    _set_graph_attributes(dot, graph_font_size)

    # Start node
    node_dim = 0.5
    dot.node(
        "start", label="", shape="circle", style="filled", fillcolor="black", width=str(node_dim), height=str(node_dim)
    )
    context = GraphBuildContext(graph=dot)
    for root in nodes:
        _add_nodes(ctx=context, node=root, parent="start", depth=0, graph_font_size=graph_font_size)

    try:
        # Suppress Graphviz warnings/errors to devnull because it clutters the terminal and things are getting properly rendered
        with open(os.devnull, "w", encoding="utf-8") as fnull:
            with redirect_stderr(fnull):
                dot.render(output_path, cleanup=True)
    except graphviz.backend.execute.ExecutableNotFound:
        raise RuntimeError(
            "Graphviz 'dot' executable not found. Ensure Graphviz is installed and in your system's PATH."
        )


def _set_graph_attributes(dot: graphviz.Digraph, graph_font_size: int = 12) -> None:
    """Sets default attributes for the Graphviz graph, nodes, and edges.

    Args:
        dot: Graphviz Digraph object
        graph_font_size: Font size for graph text elements
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

    dot.attr(
        rankdir="LR",
        bgcolor=bgcolor,
        fontname=font,
        fontsize=graph_fontsize,
        pad="0.7",
        nodesep="0.8",
        ranksep="1.5",
        splines="curved",
        overlap="false",
        dpi="300",
        labelloc="t",
        fontcolor=fontcolor,
    )

    dot.attr(
        "node",
        shape="rectangle",
        style="filled,rounded",
        fontname=font,
        fontsize=node_fontsize,
        margin="0.2,0.15",
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


def _add_nodes(ctx: GraphBuildContext, node: FunctionalityNode, parent: str, depth: int, graph_font_size: int = 12):
    """Recursively adds nodes and edges to the graph."""
    name = node.get("name")
    if not name or name in ctx.processed_nodes:
        return

    # Build HTML label with outer brackets
    html_table = _build_label(node, graph_font_size)
    label = f"<{html_table}>"

    ctx.graph.node(name, label=label, **_get_node_style(depth))
    ctx.processed_nodes.add(name)

    if (parent, name) not in ctx.processed_edges:
        ctx.graph.edge(parent, name)
        ctx.processed_edges.add((parent, name))

    for child in node.get("children", []):
        _add_nodes(ctx, child, parent=name, depth=depth + 1, graph_font_size=graph_font_size)


def _truncate_text(text: str | None, max_length: int) -> str:
    """Truncates text to a maximum length, adding ellipsis if truncated."""
    if text is None:
        return ""
    if len(text) > max_length:
        return html.escape(text[: max_length - 3].rstrip()) + "..."
    return html.escape(text)


def _build_label(node: FunctionalityNode, graph_font_size: int = 12) -> str:
    """Builds an HTML table with name, description, parameters, and outputs.

    Args:
        node: Functionality node
        graph_font_size: Font size for graph text elements
    """
    title = html.escape(node.get("name", "").replace("_", " ").title())

    # Calculate font sizes based on the graph_font_size
    title_font_size = graph_font_size + 2  # Title slightly larger
    normal_font_size = graph_font_size
    small_font_size = max(graph_font_size - 1, 8)  # Small font but not too small

    rows = [f'<tr><td><font point-size="{title_font_size}"><b>{title}</b></font></td></tr>']

    # Add node description
    description = node.get("description")
    if description:
        truncated_desc = _truncate_text(description, MAX_DESCRIPTION_LENGTH)
        rows.append(f'<tr><td><font color="#777777" point-size="{normal_font_size}"><i>{truncated_desc}</i></font></td></tr>')

    # Process Parameters
    actual_param_rows = []
    raw_params = node.get("parameters") or []

    for p_data in raw_params:
        if isinstance(p_data, dict):
            p_name = p_data.get("name")
            p_desc = p_data.get("description")
            p_options = p_data.get("options", [])  # Ensure options is a list

            # A dict parameter is significant if it has a name, or description, or non-empty options
            is_significant = bool(p_name or p_desc or (isinstance(p_options, list) and p_options))

            if is_significant:
                p_name_html = f"<b>{html.escape(p_name.replace('_', ' ').title())}</b>" if p_name else "N/A"

                if isinstance(p_options, list) and p_options:
                    options_display = [html.escape(str(opt)) for opt in p_options[:MAX_OPTIONS]]
                    options_str_intermediate = ", ".join(options_display)

                    truncated_by_count = len(p_options) > MAX_OPTIONS
                    options_str_final = ""

                    if len(options_str_intermediate) > MAX_OPTIONS_STRING_LENGTH:
                        if MAX_OPTIONS_STRING_LENGTH <= 3:
                            options_str_final = "..."
                        else:
                            options_str_final = (
                                options_str_intermediate[: MAX_OPTIONS_STRING_LENGTH - 3].rstrip().rstrip(",") + "..."
                            )
                    elif truncated_by_count:
                        options_str_final = options_str_intermediate + "..."
                    else:
                        options_str_final = options_str_intermediate
                    actual_param_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{p_name_html}: {options_str_final}</font></td></tr>')
                elif p_desc:  # Has description (name might be N/A)
                    truncated_p_desc = _truncate_text(p_desc, MAX_DESCRIPTION_LENGTH)
                    actual_param_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{p_name_html}: {truncated_p_desc}</font></td></tr>')
                elif p_name:  # Has name, but no options and no description
                    actual_param_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{p_name_html}</font></td></tr>')
        elif p_data is not None:  # Fallback for non-dict parameters, skip if None
            # Considered significant if not None and added
            actual_param_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;<b>{html.escape(str(p_data))}</b></font></td></tr>')

    if actual_param_rows:
        rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Minimal space before HR
        rows.append("<HR/>")  # Horizontal rule itself
        rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Minimal space after HR
        rows.append(f'<tr><td><font point-size="{normal_font_size}"><u>Parameters</u></font></td></tr>')
        rows.extend(actual_param_rows)

    # Process Outputs
    actual_output_rows = []
    raw_outputs = node.get("outputs") or []

    for o_data in raw_outputs:
        if isinstance(o_data, dict):
            o_category = o_data.get("category")
            o_desc = o_data.get("description")

            # An dict output is significant if it has a category or description
            is_significant = bool(o_category or o_desc)

            if is_significant:
                o_category_html = f"<b>{html.escape(o_category.replace('_', ' ').title())}</b>" if o_category else "N/A"

                if o_desc:
                    truncated_o_desc = _truncate_text(o_desc, MAX_OUTPUTS_LENGTH)
                    actual_output_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{o_category_html}: {truncated_o_desc}</font></td></tr>')
                elif o_category:  # Has category, but no description (name might be N/A)
                    actual_output_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;{o_category_html}</font></td></tr>')
        elif o_data is not None:  # Fallback for non-dict outputs, skip if None
            # Considered significant if not None and added
            actual_output_rows.append(f'<tr><td><font point-size="{normal_font_size}">&nbsp;&nbsp;<b>{html.escape(str(o_data))}</b></font></td></tr>')

    if actual_output_rows:
        rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Minimal space before HR
        rows.append("<HR/>")  # Horizontal rule itself
        rows.append('<tr><td><font point-size="1">&nbsp;</font></td></tr>')  # Minimal space after HR
        rows.append(f'<tr><td><font point-size="{normal_font_size}"><u>Outputs</u></font></td></tr>')
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

        node_name = node.get("name", "Unnamed Node")
        node_desc = node.get("description", "No description")
        f.write(f"{indent}- {node_name}: {node_desc}{param_str}{output_str}\n")

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


def write_report(
    output_dir: str,
    structured_functionalities: list[FunctionalityNode],
    limitations: list[str],
    supported_languages: list[str],
    fallback_message: str | None,
) -> None:
    """Write analysis results to a report file.

    Args:
        output_dir: Directory to write the report file to
        structured_functionalities: List of functionalities representing the workflow structure
        limitations: List of discovered limitations
        supported_languages: List of detected supported languages
        fallback_message: Detected fallback message, or None if not found
    """
    report_path = Path(output_dir) / "report.txt"

    try:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

            _write_functionalities_section(f, structured_functionalities)
            _write_json_section(f, structured_functionalities)
            _write_limitations_section(f, limitations)
            _write_languages_section(f, supported_languages)
            _write_fallback_section(f, fallback_message)

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
