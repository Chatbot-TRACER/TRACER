import sys
import yaml
import os
import re
import json
import graphviz
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam

# Import session utilities
from chatbot_explorer.session import (
    run_exploration_session,
    extract_supported_languages,
    extract_fallback_message,
    _get_all_nodes,  # Internal use likely
)
from chatbot_explorer.functionality_node import FunctionalityNode
from typing import List, Dict, Any, Set


def print_structured_functionalities(f, nodes: List[Dict[str, Any]], indent: str = ""):
    """Recursively print the structured functionalities to the file object 'f'."""
    for node in nodes:
        param_str = ""
        params_data = node.get("parameters", [])

        if isinstance(params_data, list):
            # Handle list of parameter dicts or strings
            param_str_list = []
            for p in params_data:
                if isinstance(p, dict):  # Parameter is a dict
                    param_str_list.append(p.get("name", "?"))
                elif isinstance(p, str):  # Parameter is just a string
                    param_str_list.append(p)
            if param_str_list:
                param_str = f" (Params: {', '.join(param_str_list)})"
        elif isinstance(params_data, str) and params_data.lower() not in ["none", ""]:
            # Handle parameters provided as a single string
            param_str = f" (Params: {params_data})"

        f.write(
            f"{indent}- {node.get('name', 'N/A')}: {node.get('description', 'N/A')}{param_str}\n"
        )
        if node.get("children"):
            # Recurse for children if they exist and are a list
            children_data = node.get("children", [])
            if isinstance(children_data, list):
                print_structured_functionalities(f, children_data, indent + "  ")
            else:
                # Log unexpected children type
                f.write(
                    f"{indent}  WARN: Expected 'children' to be a list, found {type(children_data)}\n"
                )


def generate_graph_image(
    structured_data: List[Dict[str, Any]], output_filename_base: str
):
    """Generates a PNG visualization of the workflow graph using Graphviz."""
    print(f"\n--- Generating workflow graph image ({output_filename_base}.png) ---")
    if not structured_data:
        print("   Skipping graph generation: No structured data provided.")
        return

    # Initialize Graphviz directed graph
    dot = graphviz.Digraph(comment="Chatbot Workflow", format="png", engine="dot")

    # Global graph attributes (styling)
    dot.attr(
        rankdir="LR",
        bgcolor="#ffffff",  # White background
        fontname="Helvetica Neue, Helvetica, Arial, sans-serif",
        fontsize="12",
        pad="0.5",
        nodesep="0.6",
        ranksep="1.2",
        splines="curved",
        overlap="false",
        dpi="300",
    )

    # Default node attributes (styling)
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

    # Default edge attributes (styling)
    dot.attr(
        "edge",
        color="#adb5bd",
        penwidth="1.0",
        arrowsize="0.7",
    )

    processed_nodes: Set[str] = set()
    processed_edges: Set[tuple[str, str]] = set()

    def add_nodes_edges(graph: graphviz.Digraph, node_dict: Dict[str, Any], depth=0):
        """Recursive helper to add nodes and edges to the graph."""
        node_name = node_dict.get("name")
        if not node_name:
            return  # Skip nodes without names

        if node_name not in processed_nodes:
            params_label = ""
            params_data = node_dict.get("parameters", [])

            # Format parameters for display in the node label
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

            # Node label text
            label = f"{node_name.replace('_', ' ')}{params_label}"

            # Node color based on depth (root, level 1, level 2, etc.)
            color_schemes = {
                0: {  # Root nodes - blue
                    "fillcolor": "#e6f3ff:#c2e0ff",
                    "color": "#4a86e8",
                },
                1: {  # First level - green
                    "fillcolor": "#e9f7ed:#c5e9d3",
                    "color": "#43a047",
                },
                2: {  # Second level - orange
                    "fillcolor": "#fef8e3:#faecc5",
                    "color": "#f6b26b",
                },
                3: {  # Third level+ - red
                    "fillcolor": "#f9e4e8:#f4c7d0",
                    "color": "#cc4125",
                },
            }

            # Use depth for color, capped at level 3 scheme
            depth_mod = min(depth, 3)
            node_style = color_schemes[depth_mod]

            # Add the node to the graph with styling
            graph.node(
                node_name,
                label=label,
                fillcolor=node_style["fillcolor"],
                color=node_style["color"],
            )

            processed_nodes.add(node_name)

        # Recursively process children
        children = node_dict.get("children", [])
        if isinstance(children, list):
            for child_dict in children:
                child_name = child_dict.get("name")
                if child_name:
                    add_nodes_edges(graph, child_dict, depth + 1)

                    # Add edge if it hasn't been added already
                    edge_key = (node_name, child_name)
                    if edge_key not in processed_edges:
                        graph.edge(node_name, child_name)
                        processed_edges.add(edge_key)
        else:
            # Log unexpected children type
            print(
                f"WARN in graph: Expected 'children' for node '{node_name}' to be a list, found {type(children)}"
            )

    # Graph title
    dot.attr(
        "graph", label="Chatbot Functionality Workflow", fontsize="18", labelloc="t"
    )

    # Start processing from root nodes
    for root_node_dict in structured_data:
        add_nodes_edges(dot, root_node_dict)

    try:
        # Render the graph to a file
        dot.render(output_filename_base, cleanup=True, view=False)
        print(f"   Successfully generated graph image: {output_filename_base}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        # Handle missing Graphviz executable
        print("\n   ERROR: Graphviz executable not found.")
        print("   Please install Graphviz (see https://graphviz.org/download/)")
        print("   and ensure it's in your system's PATH.")
    except Exception as e:
        # Handle other rendering errors
        print(f"\n   ERROR: Failed to generate graph image: {e}")


def write_report(output_dir, result, supported_languages, fallback_message):
    """Writes the analysis results to report.txt."""
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

        f.write("## FUNCTIONALITIES (Workflow Structure)\n")
        # Expecting List[Dict[str, Any]] representing the roots
        structured_functionalities = result.get("discovered_functionalities", [])

        if structured_functionalities:
            # Verify structure before printing
            if isinstance(structured_functionalities, list) and (
                not structured_functionalities
                or isinstance(structured_functionalities[0], dict)
            ):
                print_structured_functionalities(f, structured_functionalities)
            else:
                # Log if structure is unexpected
                f.write("Functionality structure not in expected dictionary format.\n")
                f.write(
                    f"Raw data:\n{json.dumps(structured_functionalities, indent=2)}\n"
                )
        else:
            f.write("No functionalities structure discovered.\n")

        # Include raw JSON for debugging/details
        f.write("\n## FUNCTIONALITIES (Raw JSON Structure)\n")
        if structured_functionalities:
            f.write(
                json.dumps(structured_functionalities, indent=2, ensure_ascii=False)
            )
        else:
            f.write("N/A\n")

        f.write("\n## LIMITATIONS\n")
        if "discovered_limitations" in result and result["discovered_limitations"]:
            for i, limitation in enumerate(result["discovered_limitations"], 1):
                f.write(f"{i}. {limitation}\n")
        else:
            f.write("No limitations discovered.\n")

        f.write("\n## SUPPORTED LANGUAGES\n")
        if supported_languages:
            for i, lang in enumerate(supported_languages, 1):
                f.write(f"{i}. {lang}\n")
        else:
            f.write("No specific language support detected.\n")

        f.write("\n## FALLBACK MESSAGE\n")
        f.write(
            fallback_message if fallback_message else "No fallback message detected.\n"
        )


def main():
    # Parse command-line arguments
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Extract arguments for clarity
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_dir = args.output
    technology = args.technology

    # Validate technology choice
    if args.technology not in valid_technologies:
        print(
            f"Invalid technology: {args.technology}. Must be one of: {valid_technologies}"
        )
        sys.exit(1)

    # Print configuration summary
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {args.technology}")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output}")
    print("====================================")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the main explorer class
    explorer = ChatbotExplorer(model_name)

    # Store results from multiple sessions
    conversation_sessions = []
    supported_languages = []
    fallback_message = None

    # Track graph exploration state
    root_nodes = []  # List of root FunctionalityNode objects
    pending_nodes = []  # Queue of nodes remaining to explore
    explored_nodes = set()  # Set of node names already explored

    # Instantiate the correct chatbot connector based on technology
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)
    else:
        # This check is redundant due to earlier validation, but good practice
        print(f"Error: Unknown technology '{args.technology}'")
        sys.exit(1)

    # --- Initial Language Detection ---
    print("\n--- Probing Chatbot Language ---")
    initial_probe_query = "Hello"  # Simple initial query
    is_ok, probe_response = the_chatbot.execute_with_input(initial_probe_query)
    if is_ok and probe_response:
        print(f"   Initial response received: '{probe_response[:60]}...'")
        try:
            # Attempt language detection via LLM
            detected_langs = extract_supported_languages(probe_response, explorer.llm)
            if detected_langs:
                supported_languages = detected_langs
                print(f"   Detected initial language(s): {supported_languages}")
            else:
                print(
                    "   Could not detect language from initial probe, defaulting to English."
                )
        except Exception as lang_e:
            print(
                f"   Error during initial language detection: {lang_e}. Defaulting to English."
            )
    else:
        print(
            "   Could not get initial response from chatbot for language probe. Defaulting to English."
        )
    # --- End Initial Language Detection ---

    # --- Exploration Loop ---
    session_num = 0  # Session counter (0-based)
    while session_num < args.sessions:
        current_session_index = session_num  # For logging (Session 1/N, etc.)
        print(f"\n=== Starting Session {current_session_index + 1}/{args.sessions} ===")

        explore_node = None  # Node to focus on this session (if any)
        session_type_log = "General Exploration"

        if pending_nodes:
            # Prioritize exploring specific nodes from the queue
            explore_node = pending_nodes.pop(0)
            # Double-check if node was already explored (shouldn't happen with current logic, but safe)
            if explore_node.name in explored_nodes:
                print(f"--- Skipping already explored node: '{explore_node.name}' ---")
                session_num += 1  # Consume a session slot
                continue
            session_type_log = f"Exploring functionality '{explore_node.name}'"
        elif (
            session_num > 0
        ):  # If queue is empty after session 0, perform general exploration
            print("   Pending nodes queue is empty. Performing general exploration.")
        # Else: Session 0 and queue is empty is the initial state.

        print(f"   Session Type: {session_type_log}")

        # Execute one exploration session
        (
            conversation_history,
            new_languages,  # Language detected this session (only used after session 0)
            new_nodes_raw,  # Raw functionality data extracted this session
            new_fallback,  # Fallback detected this session (only used after session 0)
            updated_roots,  # Updated list of root nodes
            updated_pending,  # Updated queue of nodes to explore
            updated_explored,  # Updated set of explored node names
        ) = run_exploration_session(
            current_session_index,  # Pass 0-based index
            args.sessions,
            args.turns,
            explorer,
            the_chatbot,
            current_node=explore_node,  # None for general exploration
            root_nodes=root_nodes,
            pending_nodes=pending_nodes,  # Pass current queue
            explored_nodes=explored_nodes,
            supported_languages=supported_languages,  # Pass current languages
        )

        # Aggregate results
        conversation_sessions.append(conversation_history)
        root_nodes = updated_roots
        pending_nodes = updated_pending  # Includes newly discovered nodes
        explored_nodes = updated_explored

        # Update language/fallback only after the first session (index 0)
        # Assumes these are unlikely to change and avoids repeated LLM calls.
        if current_session_index == 0:
            if new_languages:  # Update if detection was successful
                supported_languages = new_languages
                print(
                    f"   Confirmed/Updated supported languages: {supported_languages}"
                )
            if new_fallback:
                fallback_message = new_fallback
                print(f"   Confirmed fallback message.")

        # Move to the next session
        session_num += 1
    # --- End Exploration Loop ---

    # --- Post-Exploration Summary ---
    if session_num == args.sessions:
        print(f"\n=== Completed {args.sessions} exploration sessions ===")
        if pending_nodes:
            print(
                f"   NOTE: {len(pending_nodes)} nodes still remain in the pending queue."
            )
        else:
            print("   All discovered nodes were explored.")
    else:
        # Should not be reachable with the current loop structure
        print(
            f"\n--- WARNING: Exploration stopped unexpectedly after {session_num} sessions. ---"
        )

    print(f"Discovered {len(root_nodes)} root functionalities after exploration.")

    # Convert FunctionalityNode objects to dictionaries for analysis
    functionality_dicts = [node.to_dict() for node in root_nodes]

    # Prepare state for LangGraph analysis (structure inference and profile generation)
    print(
        "\n--- Preparing to infer complete workflow structure and generate user profiles ---"
    )
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            },
        ],
        "conversation_history": conversation_sessions,
        "discovered_functionalities": functionality_dicts,  # Initial structure from exploration
        "discovered_limitations": [],  # Limitations are not currently extracted during exploration
        "current_session": max_sessions,
        "exploration_finished": True,
        "conversation_goals": [],  # Not used in this flow currently
        "supported_languages": supported_languages,
        "fallback_message": fallback_message,
    }

    # 1. Infer the workflow structure using the dedicated graph
    print("\n--- Running workflow structure inference ---")
    structure_graph = explorer._build_structure_graph()
    structure_result = structure_graph.invoke(
        analysis_state, config={"configurable": {"thread_id": "structure_analysis"}}
    )

    # Use the refined structure from the structure graph for profile generation
    analysis_state["discovered_functionalities"] = structure_result[
        "discovered_functionalities"
    ]

    # 2. Generate user profiles based on the inferred structure
    print("\n--- Generating user profiles ---")
    profile_graph = explorer._build_profile_generation_graph()
    result = profile_graph.invoke(
        analysis_state, config={"configurable": {"thread_id": "analysis_session"}}
    )

    # Update functionality_dicts with the final, structured version for reporting
    functionality_dicts = structure_result.get("discovered_functionalities", [])

    # --- Save Generated Profiles ---
    built_profiles = result.get("built_profiles", [])
    if built_profiles:
        print(f"\n--- Saving {len(built_profiles)} user profiles to disk ---")

        profiles_dir = output_dir  # Save profiles in the main output directory
        os.makedirs(profiles_dir, exist_ok=True)

        for profile in built_profiles:
            # Generate a safe filename from the test_name
            test_name = profile["test_name"]

            if isinstance(test_name, dict):
                # Handle structured test names (e.g., from random generation)
                if (
                    test_name.get("function") == "random()"
                    and "data" in test_name
                    and test_name["data"]
                ):
                    # Use the first data element for a more descriptive random name
                    base_name = str(test_name["data"][0])
                    filename = (
                        f"random_profile_{base_name.lower().replace(' ', '_')}.yaml"
                    )
                else:
                    # Fallback for other dict structures
                    filename = f"profile_{hash(str(test_name))}.yaml"
            else:
                # Sanitize string test names for filenames
                filename = (
                    str(test_name)
                    .lower()
                    .replace(" ", "_")
                    .replace(",", "")
                    .replace("&", "and")
                    + ".yaml"
                )

            filepath = os.path.join(profiles_dir, filename)
            with open(filepath, "w", encoding="utf-8") as yf:
                yaml.dump(profile, yf, sort_keys=False, allow_unicode=True)
            print(f"  Saved profile: {filename}")

    # --- Write Final Report ---
    print("\n--- Writing report to disk ---")
    write_report(
        output_dir,
        # Pass the final structured functionalities and other collected data
        {"discovered_functionalities": functionality_dicts},
        supported_languages,
        fallback_message,
    )

    # --- Generate Workflow Graph Image ---
    graph_output_base = os.path.join(output_dir, "workflow_graph")
    generate_graph_image(functionality_dicts, graph_output_base)

    print("\n--- Exploration and Analysis Complete ---")


if __name__ == "__main__":
    main()
