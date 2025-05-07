"""Utilities for generating reports and visualizations from chatbot exploration."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import graphviz
import yaml

from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()


# --------------------------------------------------- #
# ---------------------- GRAPH ---------------------- #
# --------------------------------------------------- #


@dataclass
class GraphBuildContext:
    """Context holding the state during graph construction."""

    graph: graphviz.Digraph
    processed_nodes: set[str] = field(default_factory=set)
    processed_edges: set[tuple[str, str]] = field(default_factory=set)


def _set_graph_attributes(dot: graphviz.Digraph) -> None:
    """Sets default attributes for the Graphviz graph, nodes, and edges."""
    # Clean modern style
    bgcolor = "#ffffff"
    fontcolor = "#333333"
    edge_color = "#9DB2BF"
    font = "Helvetica Neue, Helvetica, Arial, sans-serif"

    dot.attr(
        rankdir="LR",
        bgcolor=bgcolor,
        fontname=font,
        fontsize="13",
        pad="0.7",
        nodesep="0.8",
        ranksep="1.5",
        splines="curved",  # Curved lines
        overlap="false",
        dpi="300",
        label="Chatbot Functionality Workflow",
        labelloc="t",
        fontcolor=fontcolor,
    )

    dot.attr(
        "node",
        shape="rectangle",
        style="filled,rounded",
        fontname=font,
        fontsize="12",
        margin="0.2,0.15",
        penwidth="1.5",  # Slightly thicker borders for modern look
        fontcolor=fontcolor,
        height="0",  # Allow height to be determined by content
        width="0",  # Allow width to be determined by content
    )

    dot.attr(
        "edge",
        color=edge_color,
        penwidth="1.2",  # Slightly thicker for visibility
        arrowsize="0.8",
        arrowhead="normal",
    )


def _get_node_style(depth: int) -> dict[str, str]:
    """Determines the node style based on its depth in the graph.

    Extended to support unlimited depth levels with a consistent color pattern.
    """
    # Modern gradient color schemes with extended palette
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


def _create_node_html_label(node_name: str, node_dict: dict, node_style: dict[str, str]) -> str:
    """Creates an HTML-like label for structured node display."""
    params_data = node_dict.get("parameters", [])
    outputs_data = node_dict.get("outputs", [])

    # If no parameters and no outputs, use a simple label
    if (not isinstance(params_data, list) or not params_data) and (
        not isinstance(outputs_data, list) or not outputs_data
    ):
        return None

    # Format node name with better spacing and capitalization
    formatted_name = node_name.replace("_", " ").title()

    # Create a structured HTML table with modern styling
    html_label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" STYLE="rounded">

    <!-- Header with gradient background -->
    <TR>
        <TD BGCOLOR="{node_style["fillcolor"].split(":")[0]}"
            STYLE="rounded"
            COLSPAN="1"
            PORT="main"
            BORDER="1"
            COLOR="{node_style["color"]}">
            <B>{formatted_name}</B>
        </TD>
    </TR>"""

    # Parameters section (if parameters exist)
    if isinstance(params_data, list) and params_data:
        html_label += f"""
        <!-- Parameters section with subtle styling -->
        <TR>
            <TD BGCOLOR="#f8f9fa"
                BORDER="1"
                COLOR="{node_style["color"]}"
                STYLE="rounded"
                ALIGN="CENTER">
                <FONT POINT-SIZE="10" COLOR="#6c757d">Parameters</FONT>
            </TD>
        </TR>

        <!-- Parameters content -->
        <TR>
            <TD BGCOLOR="#ffffff"
                ALIGN="LEFT"
                BORDER="1"
                COLOR="{node_style["color"]}"
                STYLE="rounded">
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">"""

        # Add each parameter
        for param in params_data:
            if isinstance(param, dict):
                p_name = param.get("name", "?")
                html_label += f"""
                    <TR>
                        <TD ALIGN="LEFT" BORDER="0">
                            <FONT POINT-SIZE="11"><B>{p_name}</B></FONT>
                        </TD>
                    </TR>"""

                # Add options
                p_options = param.get("options", [])
                if p_options:
                    html_label += """
                    <TR>
                        <TD ALIGN="LEFT">
                            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2" BGCOLOR="#f8f9fa">"""

                    for option in p_options:
                        html_label += f"""
                                <TR>
                                    <TD ALIGN="LEFT" CELLPADDING="1">
                                        <FONT POINT-SIZE="10" COLOR="#495057">• {option}</FONT>
                                    </TD>
                                </TR>"""

                    html_label += """
                            </TABLE>
                        </TD>
                    </TR>"""
            elif isinstance(param, str):
                html_label += f"""
                    <TR>
                        <TD ALIGN="LEFT">
                            <FONT POINT-SIZE="11">{param}</FONT>
                        </TD>
                    </TR>"""

        html_label += """
                </TABLE>
            </TD>
        </TR>"""

    # Outputs section (if outputs exist)
    if isinstance(outputs_data, list) and outputs_data:
        html_label += f"""
        <!-- Outputs section with subtle styling -->
        <TR>
            <TD BGCOLOR="#f2f7ed"
                BORDER="1"
                COLOR="{node_style["color"]}"
                STYLE="rounded"
                ALIGN="CENTER">
                <FONT POINT-SIZE="10" COLOR="#4f6e48">Outputs</FONT>
            </TD>
        </TR>

        <!-- Outputs content -->
        <TR>
            <TD BGCOLOR="#ffffff"
                ALIGN="LEFT"
                BORDER="1"
                COLOR="{node_style["color"]}"
                STYLE="rounded">
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">"""

        # Add each output category
        for output in outputs_data:
            if isinstance(output, dict):
                o_category = output.get("category", "?")
                o_desc = output.get("description", "")
                html_label += f"""
                    <TR>
                        <TD ALIGN="LEFT" BORDER="0">
                            <FONT POINT-SIZE="11"><B>{o_category}</B></FONT>
                        </TD>
                    </TR>"""

                # Add description
                if o_desc:
                    html_label += f"""
                    <TR>
                        <TD ALIGN="LEFT">
                            <FONT POINT-SIZE="10" COLOR="#4f6e48">{o_desc}</FONT>
                        </TD>
                    </TR>"""
            elif isinstance(output, str):
                html_label += f"""
                    <TR>
                        <TD ALIGN="LEFT">
                            <FONT POINT-SIZE="11">{output}</FONT>
                        </TD>
                    </TR>"""

        html_label += """
                </TABLE>
            </TD>
        </TR>"""

    html_label += """
</TABLE>>"""

    return html_label


def _add_node_to_graph(context: GraphBuildContext, node_dict: FunctionalityNode, depth: int) -> str | None:
    """Adds a single node to the graph if not already processed."""
    node_name = node_dict.get("name")
    if not node_name or node_name in context.processed_nodes:
        return node_name

    node_style = _get_node_style(depth)
    html_label = _create_node_html_label(node_name, node_dict, node_style)

    if html_label:
        context.graph.node(
            node_name,
            label=html_label,
            shape="none",  # No shape border when using HTML
            margin="0",
            color=node_style["color"],
        )
    else:
        # Simple node without parameters
        formatted_name = node_name.replace("_", " ").title()
        context.graph.node(
            node_name,
            label=formatted_name,
            fillcolor=node_style["fillcolor"],
            color=node_style["color"],
            style="filled,rounded",
            fontsize="12",
        )

    context.processed_nodes.add(node_name)
    return node_name


def _add_edges_for_children(
    context: GraphBuildContext,
    parent_name: str,
    node_dict: FunctionalityNode,
    depth: int,
) -> None:
    """Recursively adds child nodes and edges with improved styling."""
    children = node_dict.get("children", [])
    if isinstance(children, list):
        for i, child_dict in enumerate(children):
            # Recursively add child node and its subtree, passing context
            child_name = _add_nodes_and_edges_recursive(context, child_dict, depth + 1)
            if child_name:
                edge_key = (parent_name, child_name)
                if edge_key not in context.processed_edges:
                    parent_params = node_dict.get("parameters", [])
                    parent_outputs = node_dict.get("outputs", [])
                    child_params = child_dict.get("parameters", [])
                    child_outputs = child_dict.get("outputs", [])

                    parent_has_html = (isinstance(parent_params, list) and parent_params) or (
                        isinstance(parent_outputs, list) and parent_outputs
                    )
                    child_has_html = (isinstance(child_params, list) and child_params) or (
                        isinstance(child_outputs, list) and child_outputs
                    )

                    source = f"{parent_name}:main" if parent_has_html else parent_name
                    target = f"{child_name}:main" if child_has_html else child_name

                    edge_style = {}
                    if depth > 3:
                        edge_style["penwidth"] = str(1.0 + (depth % 3) * 0.1)

                    context.graph.edge(source, target, **edge_style)
                    context.processed_edges.add(edge_key)
    elif children:
        logger.warning("Expected 'children' for node '%s' to be a list, found %s", parent_name, type(children))


def _add_nodes_and_edges_recursive(
    context: GraphBuildContext,
    node_dict: FunctionalityNode,
    depth: int,
) -> str | None:
    """Recursive helper to add a node and its children to the graph."""
    # Add the current node using context
    node_name = _add_node_to_graph(context, node_dict, depth)
    if not node_name:
        return None

    # Add edges and recursively process children using context
    _add_edges_for_children(context, node_name, node_dict, depth)
    return node_name


def _create_start_node(context: GraphBuildContext) -> str:
    """Creates a simple black dot start node."""
    start_node_name = "start_node"

    context.graph.node(
        start_node_name,
        label="",
        shape="circle",
        width="0.3",
        height="0.3",
        style="filled",
        fillcolor="black",
        color="black",
    )

    context.processed_nodes.add(start_node_name)
    return start_node_name


def _render_graph(dot: graphviz.Digraph, output_filename_base: str) -> None:
    """Renders the graph to a file, handling potential errors."""
    try:
        # Add format options for higher quality output
        dot.render(
            output_filename_base,
            cleanup=True,
            view=False,
            format="png",
        )
        logger.info("Generated graph image: %s.png", output_filename_base)
    except graphviz.backend.execute.ExecutableNotFound:
        logger.exception("Graphviz executable not found. Please install Graphviz (https://graphviz.org/download/)")


def generate_graph_image(structured_data: list[FunctionalityNode], output_filename_base: str) -> None:
    """Generates a PNG visualization of the workflow graph with enhanced styling.

    Args:
        structured_data: List of root node dictionaries representing the workflow
        output_filename_base: Base path and filename for output PNG (without extension)
    """
    if not structured_data or not isinstance(structured_data, list):
        logger.warning("Skipping graph generation: No valid structured data provided")
        return

    dot = graphviz.Digraph(comment="Chatbot Workflow", format="png", engine="dot")
    _set_graph_attributes(dot)

    # Create the context object
    context = GraphBuildContext(graph=dot)

    start_node_name = _create_start_node(context)

    # Process each root node in the structured data using context
    for i, root_node_dict in enumerate(structured_data):
        if isinstance(root_node_dict, dict):
            # Add the root node and its subtree using context
            root_node_name = _add_nodes_and_edges_recursive(context, root_node_dict, 0)

            # Add edge from start node to this root node
            if root_node_name:
                edge_key = (start_node_name, root_node_name)
                if edge_key not in context.processed_edges:
                    params_data = root_node_dict.get("parameters", [])
                    outputs_data = root_node_dict.get("outputs", [])

                    has_html = (isinstance(params_data, list) and params_data) or (
                        isinstance(outputs_data, list) and outputs_data
                    )

                    target = f"{root_node_name}:main" if has_html else root_node_name

                    context.graph.edge(start_node_name, target)
                    context.processed_edges.add(edge_key)
        else:
            logger.warning("Expected root element to be a dictionary, found %s", type(root_node_dict))

    _render_graph(context.graph, output_filename_base)


# Additional function to support different export formats
def export_graph(
    structured_data: list[FunctionalityNode],
    output_filename_base: str,
    format: str = "pdf",
) -> None:
    """Exports the workflow graph in various formats with enhanced styling.

    Particularly useful for vector formats like PDF.

    Args:
        structured_data: List of root node dictionaries representing the workflow
        output_filename_base: Base path and filename for output (without extension)
        format: Output format ('png', 'svg', 'pdf')
    """
    if not structured_data or not isinstance(structured_data, list):
        logger.warning("Skipping graph export: No valid structured data provided")
        return

    dot = graphviz.Digraph(comment="Chatbot Workflow", format=format, engine="dot")
    _set_graph_attributes(dot)

    # Create the context object
    context = GraphBuildContext(graph=dot)

    # Add a dedicated start node with simple black dot styling
    start_node_name = _create_start_node(context)

    # Process each root node in the structured data using context
    for i, root_node_dict in enumerate(structured_data):
        if isinstance(root_node_dict, dict):
            root_node_name = _add_nodes_and_edges_recursive(context, root_node_dict, 0)

            if root_node_name:
                edge_key = (start_node_name, root_node_name)
                if edge_key not in context.processed_edges:
                    params_data = root_node_dict.get("parameters", [])
                    target = (
                        f"{root_node_name}:main" if isinstance(params_data, list) and params_data else root_node_name
                    )

                    context.graph.edge(start_node_name, target)
                    context.processed_edges.add(edge_key)
        else:
            logger.warning("Expected root element to be a dictionary, found %s", type(root_node_dict))

    try:
        # Add format options for higher quality output
        dot.render(
            output_filename_base,
            cleanup=True,
            view=False,
        )
        logger.info(f"Generated graph image: {output_filename_base}.{format}")
    except graphviz.backend.execute.ExecutableNotFound:
        logger.exception("Graphviz executable not found. Please install Graphviz (https://graphviz.org/download/)")


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
