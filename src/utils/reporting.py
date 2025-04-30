import json
import os
import traceback
from typing import Any

import graphviz
import yaml

# --------------------------------------------------- #
# ---------------------- GRAPH ---------------------- #
# --------------------------------------------------- #


def generate_graph_image(structured_data: list[dict[str, Any]], output_filename_base: str):
    """Generates a PNG visualization of the workflow graph using Graphviz.

    Args:
        structured_data: A list of root node dictionaries representing the workflow.
        output_filename_base: The base path and filename for the output PNG (e.g., 'output/workflow_graph').
                               The '.png' extension will be added automatically.
    """
    print(f"\n--- Generating workflow graph image ({output_filename_base}.png) ---")
    if not structured_data:
        print("   Skipping graph generation: No structured data provided.")
        return

    dot = graphviz.Digraph(comment="Chatbot Workflow", format="png", engine="dot")
    dot.attr(
        rankdir="LR",
        bgcolor="#ffffff",
        fontname="Helvetica Neue, Helvetica, Arial, sans-serif",
        fontsize="12",
        pad="0.5",
        nodesep="0.6",
        ranksep="1.2",
        splines="curved",
        overlap="false",
        dpi="300",
    )
    dot.attr(
        "node",
        shape="rectangle",
        style="filled,rounded",
        fontname="Helvetica Neue, Helvetica, Arial, sans-serif",
        fontsize="10",
        margin="0.15,0.1",
        penwidth="1.0",
        fontcolor="#333333",
    )
    dot.attr(
        "edge",
        color="#adb5bd",
        penwidth="1.0",
        arrowsize="0.7",
    )

    processed_nodes: set[str] = set()
    processed_edges: set[tuple[str, str]] = set()

    def add_nodes_edges(graph: graphviz.Digraph, node_dict: dict[str, Any], depth=0):
        """Recursive helper to add nodes and edges to the graph."""
        node_name = node_dict.get("name")
        if not node_name:
            return

        if node_name not in processed_nodes:
            params_label = ""
            params_data = node_dict.get("parameters", [])
            if isinstance(params_data, list):
                param_str_list = []
                for p in params_data:
                    if isinstance(p, dict):
                        param_str_list.append(p.get("name", "?"))
                    elif isinstance(p, str):
                        param_str_list.append(p)
                if param_str_list:
                    params_label = f"\nParams: {', '.join(param_str_list)}"
            elif isinstance(params_data, str) and params_data.lower() not in [
                "none",
                "",
            ]:
                params_label = f"\nParams: {params_data}"

            label = f"{node_name.replace('_', ' ')}{params_label}"
            color_schemes = {
                0: {"fillcolor": "#e6f3ff:#c2e0ff", "color": "#4a86e8"},
                1: {"fillcolor": "#e9f7ed:#c5e9d3", "color": "#43a047"},
                2: {"fillcolor": "#fef8e3:#faecc5", "color": "#f6b26b"},
                3: {"fillcolor": "#f9e4e8:#f4c7d0", "color": "#cc4125"},
            }
            depth_mod = min(depth, 3)
            node_style = color_schemes[depth_mod]

            graph.node(
                node_name,
                label=label,
                fillcolor=node_style["fillcolor"],
                color=node_style["color"],
            )
            processed_nodes.add(node_name)

        children = node_dict.get("children", [])
        if isinstance(children, list):
            for child_dict in children:
                child_name = child_dict.get("name")
                if child_name:
                    add_nodes_edges(graph, child_dict, depth + 1)
                    edge_key = (node_name, child_name)
                    if edge_key not in processed_edges:
                        graph.edge(node_name, child_name)
                        processed_edges.add(edge_key)
        elif children:
            print(f"WARN in graph: Expected 'children' for node '{node_name}' to be a list, found {type(children)}")

    dot.attr("graph", label="Chatbot Functionality Workflow", fontsize="18", labelloc="t")
    start_node_name = "start_node"
    dot.node(
        start_node_name,
        label="",
        shape="circle",
        width="0.3",
        height="0.3",
        style="filled",
        fillcolor="black",
        color="black",
    )

    for root_node_dict in structured_data:
        node_name = root_node_dict.get("name")
        if node_name:
            add_nodes_edges(dot, root_node_dict)
            edge_key = (start_node_name, node_name)
            if edge_key not in processed_edges:
                dot.edge(start_node_name, node_name)
                processed_edges.add(edge_key)

    try:
        dot.render(output_filename_base, cleanup=True, view=False)
        print(f"   Successfully generated graph image: {output_filename_base}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n   ERROR: Graphviz executable not found.")
        print("   Please install Graphviz (see https://graphviz.org/download/)")
        print("   and ensure it's in your system's PATH.")
    except Exception as e:
        print(f"\n   ERROR: Failed to generate graph image: {e}")


# ------------------------------------------------ #
# -------------------- REPORT -------------------- #
# ------------------------------------------------ #


def print_structured_functionalities(f, nodes: list[dict[str, Any]], indent: str = ""):
    """Recursively print the structured functionalities to a file object.

    Args:
        f: The file object to write to.
        nodes: A list of node dictionaries representing the functionalities.
        indent: The string used for indentation (e.g., "  ").
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
                    p_type = param.get("type", "N/A")
                    p_desc = param.get("description", "N/A")
                    param_details.append(f"{p_name} ({p_type}): {p_desc}")
                else:
                    param_details.append(f"InvalidParamFormat({type(param)})")
            param_str = f" | Params: [{'; '.join(param_details)}]"
        elif params_data:  # If not a list but exists
            param_str = f" | Params: InvalidFormat({type(params_data)})"

        node_name = node.get("name", "Unnamed Node")
        node_desc = node.get("description", "No description")
        f.write(f"{indent}- {node_name}: {node_desc}{param_str}\n")

        children = node.get("children", [])
        if children and isinstance(children, list):
            # Recursive call - ensure children is also List[Dict] before calling
            if all(isinstance(child, dict) for child in children):
                print_structured_functionalities(f, children, indent + "  ")
            else:
                f.write(f"{indent}  ERROR: Children of '{node_name}' contains non-dictionary elements.\n")
        elif children:  # If not a list but exists
            f.write(f"{indent}  ERROR: Children of '{node_name}' is not a list ({type(children)}).\n")


def write_report(
    output_dir: str,
    structured_functionalities: list[dict[str, Any]],
    limitations: list[str],
    supported_languages: list[str],
    fallback_message: str | None,
):
    """Writes the analysis results to report.txt.

    Args:
        output_dir: The directory to write the report file to.
        structured_functionalities: A list of dictionaries representing the workflow structure.
        limitations: A list of discovered limitations strings.
        supported_languages: A list of detected supported languages.
        fallback_message: The detected fallback message, or None.
    """
    report_path = os.path.join(output_dir, "report.txt")
    print(f"\n--- Writing analysis report ({report_path}) ---")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

            f.write("## FUNCTIONALITIES (Workflow Structure)\n")
            # Check if it's a list AND if its elements are dicts (if not empty)
            if isinstance(structured_functionalities, list):
                if structured_functionalities:
                    # Check the first element's type rigorously before calling print_structured
                    if isinstance(structured_functionalities[0], dict):
                        print_structured_functionalities(f, structured_functionalities)
                    else:
                        f.write("Functionality structure is a list, but elements are not dictionaries.\n")
                        f.write(f"First element type: {type(structured_functionalities[0])}\n")
                        try:
                            f.write(f"Raw data:\n{json.dumps(structured_functionalities, indent=2)}\n")
                        except TypeError:
                            f.write(f"Raw data (repr):\n{structured_functionalities!r}\n")
                else:
                    f.write("No functionalities structure discovered (empty list).\n")
            elif structured_functionalities is not None:  # Handle case where it's not None but not a list
                f.write("Functionality structure not in expected list format.\n")
                f.write(f"Type received: {type(structured_functionalities)}\n")
                try:
                    f.write(f"Raw data:\n{json.dumps(structured_functionalities, indent=2)}\n")
                except TypeError:
                    f.write(f"Raw data (repr):\n{structured_functionalities!r}\n")
            else:  # Handle None case
                f.write("No functionalities structure discovered (None).\n")

            f.write("\n## FUNCTIONALITIES (Raw JSON Structure)\n")
            # Check type before attempting JSON dump
            if isinstance(structured_functionalities, list):
                try:
                    f.write(json.dumps(structured_functionalities, indent=2, ensure_ascii=False))
                except TypeError as json_e:
                    f.write(f"Could not serialize functionalities to JSON: {json_e}\n")
                    f.write(f"Data (repr): {structured_functionalities!r}\n")  # Raw repr
            elif structured_functionalities is not None:
                f.write("Functionality structure not in list format, cannot dump as JSON array.\n")
                f.write(f"Raw data (repr):\n{structured_functionalities!r}\n")
            else:
                f.write("N/A\n")

            f.write("\n## LIMITATIONS\n")
            # Check if it's a list AND if its elements are strings (if not empty)
            if isinstance(limitations, list):
                if limitations:
                    # Check the first element's type rigorously
                    if isinstance(limitations[0], str):
                        for i, limitation in enumerate(limitations, 1):
                            f.write(f"{i}. {limitation}\n")
                    else:
                        f.write("Limitations list elements are not strings.\n")
                        f.write(f"First element type: {type(limitations[0])}\n")
                        try:
                            f.write(f"Raw data:\n{json.dumps(limitations, indent=2)}\n")
                        except TypeError:
                            f.write(f"Raw data (repr):\n{limitations!r}\n")
                else:
                    f.write("No limitations discovered (empty list).\n")
            elif limitations is not None:  # Handle case where it's not None but not a list
                f.write("Limitations data not in expected list format.\n")
                f.write(f"Type received: {type(limitations)}\n")
                try:
                    f.write(f"Raw data:\n{json.dumps(limitations, indent=2)}\n")
                except TypeError:
                    f.write(f"Raw data (repr):\n{limitations!r}\n")
            else:  # Handle None case
                f.write("No limitations discovered (None).\n")

            f.write("\n## SUPPORTED LANGUAGES\n")
            # Check if it's a list AND if its elements are strings (if not empty)
            if isinstance(supported_languages, list):
                if supported_languages:
                    if isinstance(supported_languages[0], str):
                        for i, lang in enumerate(supported_languages, 1):
                            f.write(f"{i}. {lang}\n")
                    else:
                        f.write("Supported languages list elements are not strings.\n")
                        f.write(f"First element type: {type(supported_languages[0])}\n")
                        try:
                            f.write(f"Raw data:\n{json.dumps(supported_languages, indent=2)}\n")
                        except TypeError:
                            f.write(f"Raw data (repr):\n{supported_languages!r}\n")
                else:
                    f.write("No specific language support detected (empty list).\n")
            elif supported_languages is not None:
                f.write("Supported languages data not in expected list format.\n")
                f.write(f"Type received: {type(supported_languages)}\n")
                try:
                    f.write(f"Raw data:\n{json.dumps(supported_languages, indent=2)}\n")
                except TypeError:
                    f.write(f"Raw data (repr):\n{supported_languages!r}\n")
            else:
                f.write("No specific language support detected (None).\n")

            f.write("\n## FALLBACK MESSAGE\n")
            # Check type before writing
            if isinstance(fallback_message, str):
                f.write(fallback_message)
            elif fallback_message is None:
                f.write("No fallback message detected.")
            else:
                f.write("Fallback message data is not a string or None.\n")
                f.write(f"Type received: {type(fallback_message)}\n")
                f.write(f"Raw data: {fallback_message!r}\n")

        print(f"   Successfully wrote report: {report_path}")
    except OSError as e:
        print(f"   ERROR: Failed to write report file: {e}")
    except Exception as e:
        print(f"   ERROR: An unexpected error occurred while writing report: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")


# -------------------------------------------------- #
# -------------------- PROFILES -------------------- #
# -------------------------------------------------- #


def save_profiles(built_profiles: list[dict[str, Any]], output_dir: str):
    """Saves the generated user profiles to individual YAML files in the specified directory.

    Args:
        built_profiles: A list of dictionaries, where each dictionary represents a user profile.
                        Expected keys: 'test_name', and the rest of the profile content.
        output_dir: The directory to write the profile files to.
    """
    if not built_profiles:
        print("\n--- Skipping profile saving: No profiles generated ---")
        return

    print(f"\n--- Saving {len(built_profiles)} user profiles to disk ({output_dir}) ---")

    # Ensure the output directory exists (it might already exist from main.py, but double-check)
    os.makedirs(output_dir, exist_ok=True)

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

        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as yf:
                # Dump the entire profile dictionary
                yaml.dump(
                    profile,
                    yf,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=1000,
                )
            # print(f"  Saved profile: {filename}") # Keep output less verbose
            saved_count += 1
        except yaml.YAMLError as e:
            print(f"   ERROR: Failed to dump YAML for profile '{test_name}': {e}")
            error_count += 1
        except OSError as e:
            print(f"   ERROR: Failed to write file '{filename}': {e}")
            error_count += 1
        except Exception as e:
            print(f"   ERROR: An unexpected error occurred while writing profile '{test_name}': {e}")
            error_count += 1

    print(f"   Finished saving profiles: {saved_count} successful, {error_count} errors.")
