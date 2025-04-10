import sys
import yaml
import os
import re
import json
import graphviz
from chatbot_explorer.cli import parse_arguments
from chatbot_explorer.explorer import ChatbotExplorer
from chatbot_connectors import ChatbotTaskyto, ChatbotAdaUam
from chatbot_explorer.session import run_exploration_session
from chatbot_explorer.functionality_node import FunctionalityNode
from typing import List, Dict, Any, Set


def print_structured_functionalities(f, nodes: List[Dict[str, Any]], indent: str = ""):
    """
    Helper function to recursively print the structured functionalities.
    Handles variations in the 'parameters' field type.
    """
    for node in nodes:
        param_str = ""
        params_data = node.get("parameters", [])

        if isinstance(params_data, list):
            # Process as list of dicts
            param_str_list = []
            for p in params_data:
                if isinstance(p, dict):  # Ensure item in list is a dict
                    param_str_list.append(p.get("name", "?"))
                elif isinstance(p, str):  # Handle case where list contains strings
                    param_str_list.append(p)  # Just use the string
            if param_str_list:
                param_str = f" (Params: {', '.join(param_str_list)})"
        elif isinstance(params_data, str) and params_data.lower() not in ["none", ""]:
            # Handle case where parameters is just a string
            param_str = f" (Params: {params_data})"

        f.write(
            f"{indent}- {node.get('name', 'N/A')}: {node.get('description', 'N/A')}{param_str}\n"
        )
        if node.get("children"):
            # Ensure children is also a list before recursing
            children_data = node.get("children", [])
            if isinstance(children_data, list):
                print_structured_functionalities(f, children_data, indent + "  ")
            else:
                f.write(
                    f"{indent}  WARN: Expected 'children' to be a list, found {type(children_data)}\n"
                )


def generate_graph_image(
    structured_data: List[Dict[str, Any]], output_filename_base: str
):
    """
    Generates a visually appealing PNG image visualizing the workflow graph using Graphviz.

    Args:
        structured_data: List of root node dictionaries representing the workflow.
        output_filename_base: The base path and filename for the output image.
    """
    print(f"\n--- Generating workflow graph image ({output_filename_base}.png) ---")
    if not structured_data:
        print("   Skipping graph generation: No structured data provided.")
        return

    # Initialize a directed graph with modern styling
    dot = graphviz.Digraph(comment="Chatbot Workflow", format="png", engine="dot")

    # Set global graph attributes for better aesthetics
    dot.attr(
        rankdir="LR",
        bgcolor="#ffffff",  # Can be changed to transparent
        fontname="Helvetica Neue, Helvetica, Arial, sans-serif",
        fontsize="12",
        pad="0.5",
        nodesep="0.6",
        ranksep="1.2",
        splines="curved",
        overlap="false",
        dpi="300",
    )

    # Set default node attributes
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

    # Set default edge attributes
    dot.attr(
        "edge",
        color="#adb5bd",
        penwidth="1.0",
        arrowsize="0.7",
    )

    processed_nodes: Set[str] = set()

    def add_nodes_edges(graph: graphviz.Digraph, node_dict: Dict[str, Any], depth=0):
        """Recursive helper to add nodes and edges with enhanced styling."""
        node_name = node_dict.get("name")
        if not node_name:
            return

        if node_name not in processed_nodes:
            params_label = ""
            params_data = node_dict.get("parameters", [])

            # Process parameters for display
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

            # Format node label
            label = f"{node_name.replace('_', ' ')}{params_label}"

            # Different color schemes based on node depth
            color_schemes = {
                0: {
                    "fillcolor": "#e6f3ff:#c2e0ff",
                    "color": "#4a86e8",
                },  # Root nodes - blue
                1: {
                    "fillcolor": "#e9f7ed:#c5e9d3",
                    "color": "#43a047",
                },  # First level - green
                2: {
                    "fillcolor": "#fef8e3:#faecc5",
                    "color": "#f6b26b",
                },  # Second level - orange
                3: {
                    "fillcolor": "#f9e4e8:#f4c7d0",
                    "color": "#cc4125",
                },  # Third level - red
            }

            # Cap at 3 levels of color differentiation
            depth_mod = min(depth, 3)
            node_style = color_schemes[depth_mod]

            # Apply styling and add node
            graph.node(
                node_name,
                label=label,
                fillcolor=node_style["fillcolor"],
                color=node_style["color"],
            )

            processed_nodes.add(node_name)

        # Process children with level-based styling
        children = node_dict.get("children", [])
        if isinstance(children, list):
            for child_dict in children:
                child_name = child_dict.get("name")
                if child_name:
                    add_nodes_edges(graph, child_dict, depth + 1)
                    graph.edge(node_name, child_name)
        else:
            print(
                f"WARN in graph: Expected 'children' for node '{node_name}' to be a list, found {type(children)}"
            )

    # Add title
    dot.attr(
        "graph", label="Chatbot Functionality Workflow", fontsize="18", labelloc="t"
    )

    # Process all root nodes
    for root_node_dict in structured_data:
        add_nodes_edges(dot, root_node_dict)

    try:
        dot.render(output_filename_base, cleanup=True, view=False)
        print(f"   Successfully generated graph image: {output_filename_base}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n   ERROR: Graphviz executable not found.")
        print("   Please install Graphviz (see https://graphviz.org/download/)")
        print("   and ensure it's in your system's PATH.")
    except Exception as e:
        print(f"\n   ERROR: Failed to generate graph image: {e}")


def write_report(output_dir, result, supported_languages, fallback_message):
    """
    Write discovered functionalities (structured), limitations, languages, and fallback message.
    """
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=== CHATBOT FUNCTIONALITY ANALYSIS ===\n\n")

        f.write("## FUNCTIONALITIES (Workflow Structure)\n")
        # result['discovered_functionalities'] is now List[Dict[str, Any]] representing the roots
        structured_functionalities = result.get("discovered_functionalities", [])

        if structured_functionalities:
            # Check if it looks like our structured dicts
            if isinstance(structured_functionalities, list) and (
                not structured_functionalities
                or isinstance(structured_functionalities[0], dict)
            ):
                print_structured_functionalities(f, structured_functionalities)
            else:
                f.write("Functionality structure not in expected dictionary format.\n")
                f.write(
                    f"Raw data:\n{json.dumps(structured_functionalities, indent=2)}\n"
                )  # Print raw data
        else:
            f.write("No functionalities structure discovered.\n")

        # Add a raw JSON dump for detailed inspection
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
    # Parse command line arguments
    args = parse_arguments()
    valid_technologies = ["taskyto", "ada-uam"]

    # Use the parameters from args
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_dir = args.output
    technology = args.technology

    # Validate the technology argument
    if args.technology not in valid_technologies:
        print(
            f"Invalid technology: {args.technology}. Must be one of: {valid_technologies}"
        )
        sys.exit(1)

    # Display configuration
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot Technology: {args.technology}")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print(f"Output directory: {args.output}")
    print("====================================")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize explorer
    explorer = ChatbotExplorer(model_name)

    # Track multiple conversation sessions
    conversation_sessions = []
    supported_languages = []
    discovered_functionalities = []
    fallback_message = None

    # Create the chatbot according to the technology
    if technology == "taskyto":
        the_chatbot = ChatbotTaskyto(chatbot_url)
    elif technology == "ada-uam":
        the_chatbot = ChatbotAdaUam(chatbot_url)

    # Run exploration sessions, each session is a conversation
    for session_num in range(max_sessions):
        # Run the exploration session
        conversation_history, new_languages, new_topics, new_fallback = (
            run_exploration_session(
                session_num,
                max_sessions,
                max_turns,
                explorer,
                the_chatbot,
                supported_languages,
                discovered_functionalities,
            )
        )

        # Store session data
        conversation_sessions.append(conversation_history)

        # Update supported languages if detected
        if session_num == 0 and new_languages:
            supported_languages = new_languages
            fallback_message = new_fallback

        # Update discovered functionalities
        for topic in new_topics:
            if topic not in discovered_functionalities:
                discovered_functionalities.append(topic)
                print(f"New functionality discovered: {topic}")

        # After session ends, save the conversation history
        print(f"\nSession {session_num + 1} complete")
        conversation_sessions.append(conversation_history)

    # Create state for analysis
    print("\n--- All exploration sessions complete. Analyzing results... ---")
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            }
        ],
        "conversation_history": conversation_sessions,
        "discovered_functionalities": [],
        "discovered_limitations": [],
        "current_session": max_sessions,
        "exploration_finished": True,
        "conversation_goals": [],
        "supported_languages": supported_languages,
        "fallback_message": fallback_message,
    }

    # Execute the analysis
    config = {"configurable": {"thread_id": "analysis_session"}}
    result = explorer.run_exploration(analysis_state, config)

    # Display brief summary of what was discovered
    print("\n=== Analysis Complete ===")
    print(f"Found {len(result.get('discovered_functionalities', []))} functionalities")
    print(f"Found {len(result.get('discovered_limitations', []))} limitations")
    print(
        f"Supported languages: {', '.join(supported_languages) if supported_languages else 'None detected'}"
    )

    # Save profiles from the built_profiles in result
    built_profiles = result.get("built_profiles", [])
    if built_profiles:
        print(f"\n--- Saving {len(built_profiles)} user profiles to disk ---")

        # If output_dir is directly specified, use it as the profile directory
        profiles_dir = output_dir
        os.makedirs(profiles_dir, exist_ok=True)

        for profile in built_profiles:
            # Handle cases where test_name might be a dict (like random profile generators)
            test_name = profile["test_name"]

            if isinstance(test_name, dict):
                # For random profiles, use a consistent filename with index
                if (
                    test_name.get("function") == "random()"
                    and "data" in test_name
                    and test_name["data"]
                ):
                    # Use the first item in the data list as a base name
                    base_name = str(test_name["data"][0])
                    filename = (
                        f"random_profile_{base_name.lower().replace(' ', '_')}.yaml"
                    )
                else:
                    # Generic fallback for other dict types
                    filename = f"profile_{hash(str(test_name))}.yaml"
            else:
                # Normal string case
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

        print(f"\nAll profiles saved to: {output_dir}")

    # --- Call write_report ---
    print("\n--- Writing report to disk ---")
    write_report(
        output_dir,
        result,
        supported_languages,
        fallback_message,
    )
    print(f"   Report saved to: {os.path.join(output_dir, 'report.txt')}")

    # --- Call generate_graph_image ---
    structured_functionalities = result.get("discovered_functionalities", [])
    if structured_functionalities:
        # Define the output path for the graph image
        graph_output_base = os.path.join(output_dir, "workflow_graph")
        generate_graph_image(structured_functionalities, graph_output_base)
    else:
        print(
            "\n--- Skipping graph generation: No structured functionalities found in results. ---"
        )

    print("\n--- Exploration and Analysis Complete ---")


if __name__ == "__main__":
    main()
