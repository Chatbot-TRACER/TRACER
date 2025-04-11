from typing import Annotated, Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import os
import re
import yaml
import json

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from .nodes.goals_node import generate_user_profiles_and_goals
from .nodes.analyzer_node import analyze_conversations
from .nodes.conversation_parameters_node import generate_conversation_parameters

from .validation_script import YamlValidator

from .functionality_node import FunctionalityNode

# Takes anything that is between exactly two curly braces
VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


class State(TypedDict):
    """State for the LangGraph flow"""

    messages: Annotated[list, add_messages]
    conversation_history: list
    discovered_functionalities: List[FunctionalityNode]
    discovered_limitations: list
    current_session: int
    exploration_finished: bool
    conversation_goals: list
    supported_languages: list
    built_profiles: list
    fallback_message: str


class ChatbotExplorer:
    """Manages exploration of target chatbots using LangGraph."""

    def __init__(self, model_name: str):
        """Initialize explorer with the given model name."""
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph for exploration."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("explorer", self._explorer_node)
        graph_builder.add_node("analyzer", self._analyzer_node)
        graph_builder.add_node("structure_builder", self._structure_builder_node)
        graph_builder.add_node("goal_generator", self._goal_generator_node)
        graph_builder.add_node("conversation_params", self._conversation_params_node)
        graph_builder.add_node("profile_builder", self._build_profiles_node)
        graph_builder.add_node("profile_validator", self._validate_profiles_node)

        # Add edges
        graph_builder.set_entry_point("explorer")
        graph_builder.add_edge("explorer", "analyzer")
        graph_builder.add_edge("analyzer", "structure_builder")
        graph_builder.add_edge("structure_builder", "goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        return graph_builder.compile(checkpointer=self.memory)

    def _build_profile_generation_graph(self):
        """Build a graph that skips analysis and structure building (for use with pre-built structures)"""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("goal_generator", self._goal_generator_node)
        graph_builder.add_node("conversation_params", self._conversation_params_node)
        graph_builder.add_node("profile_builder", self._build_profiles_node)
        graph_builder.add_node("profile_validator", self._validate_profiles_node)

        # Add edges
        graph_builder.set_entry_point("goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        return graph_builder.compile(checkpointer=self.memory)

    def _build_structure_graph(self):
        """Build a graph that only runs the structure builder node"""
        graph_builder = StateGraph(State)

        # Add structure builder node
        graph_builder.add_node("structure_builder", self._structure_builder_node)

        # Set it as both entry and finish point
        graph_builder.set_entry_point("structure_builder")
        graph_builder.set_finish_point("structure_builder")

        return graph_builder.compile(checkpointer=self.memory)

    def _explorer_node(self, state: State):
        """Explorer node for interacting with the target chatbot."""
        if not state["exploration_finished"]:
            max_history_messages = 20  # Keep last 10 turns (user + assistant)
            messages_to_send = []
            if state["messages"]:
                messages_to_send.append(state["messages"][0])  # Keep system prompt
                messages_to_send.extend(state["messages"][-max_history_messages:])

            if not messages_to_send:  # Handle empty initial state if necessary
                return {"messages": state["messages"]}  # Or handle appropriately

            return {"messages": [self.llm.invoke(messages_to_send)], "explored": True}

        return {"messages": state["messages"]}

    def _analyzer_node(self, state: State) -> State:
        """
        Analyzer node: Calls the analyzer module to get structured FunctionalityNodes
        and limitations, then updates the state.
        """
        if not state.get("exploration_finished", False):
            print("Skipping analysis node: Exploration not finished.")
            return state

        print("\n--- Analyzing conversations (using updated analyzer module) ---")

        if not state.get("conversation_history"):
            print("Warning: Cannot analyze, conversation_history is missing.")
            return {
                **state,
                "discovered_functionalities": [],
                "discovered_limitations": [],
            }

        try:
            analysis_output = analyze_conversations(
                state["conversation_history"],
                state.get("supported_languages", []),
                self.llm,
            )
            print(
                f" -> Analysis complete. Found {len(analysis_output.get('functionalities', []))} functionalities (as nodes) and {len(analysis_output.get('limitations', []))} limitations."
            )

        except Exception as e:
            print(f"Error during conversation analysis call: {e}")
            return {
                **state,
                "discovered_functionalities": [],
                "discovered_limitations": [],
            }

        # Convert FunctionalityNode objects to dict before storing
        functionality_dicts = [
            func.to_dict() for func in analysis_output.get("functionalities", [])
        ]

        # Prepare the output state dictionary
        output_state = {**state}
        output_state["messages"] = state.get("messages", []) + [
            analysis_output.get("analysis_result", "Analysis log missing.")
        ]
        # Store dicts instead of raw FunctionalityNode objects
        output_state["discovered_functionalities"] = functionality_dicts
        output_state["discovered_limitations"] = analysis_output.get("limitations", [])

        return output_state

    def _goal_generator_node(self, state: State) -> State:
        if state.get("exploration_finished", False) and state.get(
            "discovered_functionalities"
        ):
            print("\n--- Generating conversation goals from structured data ---")

            structured_root_dicts: List[Dict[str, Any]] = state[
                "discovered_functionalities"
            ]

            # --- Helper function to recursively get all descriptions ---
            def get_all_descriptions(nodes: List[Dict[str, Any]]) -> List[str]:
                descriptions = []
                for node in nodes:
                    # Add description if present
                    if node.get("description"):
                        descriptions.append(node["description"])
                    # Add description of all children
                    if node.get("children"):
                        descriptions.extend(get_all_descriptions(node["children"]))
                return descriptions

            functionality_descriptions = get_all_descriptions(structured_root_dicts)

            if not functionality_descriptions:
                print(
                    "   Warning: No descriptions found in structured functionalities."
                )
                return state

            print(
                f" -> Preparing {len(functionality_descriptions)} descriptions (from structure) for goal generation."
            )

            try:
                profiles_with_goals = generate_user_profiles_and_goals(
                    functionality_descriptions,
                    state.get("discovered_limitations", []),
                    self.llm,
                    conversation_history=state.get("conversation_history", []),
                    supported_languages=state.get("supported_languages", []),
                )
                print(f" -> Generated {len(profiles_with_goals)} profiles with goals.")
                return {**state, "conversation_goals": profiles_with_goals}

            except Exception as e:
                print(f"Error during goal generation: {e}")
                return {**state, "conversation_goals": []}

        elif state.get("exploration_finished", False):
            print("\n--- Skipping goal generation: No functionalities discovered. ---")

        return state

    def _conversation_params_node(self, state: State):
        """Node for generating conversation parameters for profiles."""
        if state["exploration_finished"] and state["conversation_goals"]:
            print("\n--- Generating conversation parameters ---")

            # The state["discovered_functionalities"] now contains the structured dicts
            structured_root_dicts = state.get("discovered_functionalities", [])

            # Flattening to pass info similar to before the dictionaries
            def get_all_func_info(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                all_info = []
                for node in nodes:
                    # Add current node info (without children for flat list)
                    info = {k: v for k, v in node.items() if k != "children"}
                    all_info.append(info)
                    if node.get("children"):
                        all_info.extend(get_all_func_info(node["children"]))
                return all_info

            flat_func_info = get_all_func_info(structured_root_dicts)

            profiles_with_params = generate_conversation_parameters(
                state["conversation_goals"],
                flat_func_info,  # Pass the flat list of info dicts
                self.llm,
                supported_languages=state.get("supported_languages", []),
            )

            return {
                "messages": state["messages"],
                "conversation_goals": profiles_with_params,
            }
        return {"messages": state["messages"]}

    def _build_profiles_node(self, state: State):
        """Node that builds YAML profiles and stores them in the state."""
        if state["exploration_finished"] and state["conversation_goals"]:
            print("\n--- Building user profiles ---")
            built_profiles = []

            # Get fallback message from state or use default
            fallback_message = state.get(
                "fallback_message", "I'm sorry, I don't understand."
            )

            # Get primary language from supported languages or default to English
            primary_language = "English"
            if (
                state.get("supported_languages")
                and len(state["supported_languages"]) > 0
            ):
                primary_language = state["supported_languages"][0]

            for profile in state["conversation_goals"]:
                profile_yaml = self._build_profile_yaml(
                    profile,
                    fallback_message=fallback_message,
                    primary_language=primary_language,
                )
                built_profiles.append(profile_yaml)
            return {"messages": state["messages"], "built_profiles": built_profiles}
        return {"messages": state["messages"]}

    def run_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Run graph with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.invoke(state, config=config)

    def stream_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream graph execution with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.stream(state, config=config)

    def _build_profile_yaml(self, profile, fallback_message, primary_language):
        """
        Build the base YAML dictionary for a given profile.
        """
        # Collect all variables used by the user goals
        used_variables = set()
        for goal in profile.get("goals", []):
            variables_in_goals = VARIABLE_PATTERN.findall(goal)
            used_variables.update(variables_in_goals)

        # Combine raw text goals and variable references
        yaml_goals = list(profile.get("goals", []))
        for var_name in used_variables:
            if var_name in profile:
                yaml_goals.append({var_name: profile[var_name]})

        # Build chatbot section
        chabot_section = {
            "is_starter": False,
            "fallback": fallback_message,
        }
        if "outputs" in profile:
            chabot_section["output"] = profile["outputs"]

        # Build user context
        user_context = ["personality: personalities/conversational-user.yml"]
        context = profile.get("context", [])
        if isinstance(context, str):
            user_context.append(context)
        else:
            for ctx_item in context:
                user_context.append(ctx_item)

        # Final conversation section
        conversation_section = profile.get("conversation", {})

        # Return the YAML dictionary
        return {
            "test_name": profile["name"],
            "llm": {
                "temperature": 0.8,
                "model": "gpt-4o-mini",
                "format": {"type": "text"},
            },
            "user": {
                "language": primary_language,
                "role": profile["role"],
                "context": user_context,
                "goals": yaml_goals,
            },
            "chatbot": chabot_section,
            "conversation": conversation_section,
        }

    def _validate_profiles_node(self, state: State):
        """Node that validates the built profiles and, if needed, asks the LLM to fix them."""
        if state["exploration_finished"] and state.get("built_profiles"):
            print("\n--- Validating user profiles ---")
            validator = YamlValidator()
            validated_profiles = []

            for profile in state["built_profiles"]:
                yaml_content = yaml.dump(profile, sort_keys=False, allow_unicode=True)
                errors = validator.validate(yaml_content)

                if not errors:
                    validated_profiles.append(profile)
                    print(f"✓ Profile '{profile['test_name']}' valid, no fixes needed.")
                else:
                    error_count = len(errors)
                    print(
                        f"\n⚠ Profile '{profile['test_name']}' has {error_count} validation errors"
                    )

                    for e in errors[:3]:
                        print(f"  • {e.path}: {e.message}")
                    if error_count > 3:
                        print(f"  • ... and {error_count - 3} more errors")

                    error_messages = "\n".join(
                        f"- {e.path}: {e.message}" for e in errors
                    )
                    fix_prompt = (
                        "You are an AI assistant specialized in correcting YAML configuration files.\n"
                        "Based ONLY on the following validation errors, please fix the provided YAML content.\n"
                        "Your response MUST contain ONLY the complete, corrected YAML content.\n"
                        "Enclose the corrected YAML within triple backticks (```yaml ... ```).\n"
                        "Do NOT include any explanations, apologies, introductions, or conclusions outside the YAML block.\n"
                        "Ensure the output is well-formed YAML and directly addresses the errors listed.\n\n"
                        f"Errors to fix:\n{error_messages}\n\n"
                        "Original YAML to fix:\n"
                        f"```yaml\n{yaml_content}\n```\n\n"
                        "Corrected YAML:"
                    )

                    print("  Asking LLM to fix the profile...")

                    try:
                        fixed_yaml_str = self.llm.invoke(
                            [{"role": "user", "content": fix_prompt}]
                        )

                        # Function to extract code fenced YAML from response
                        def _extract_yaml(text: str) -> str:
                            if hasattr(text, "content"):
                                text = text.content

                            # different patterns to extract YAML
                            yaml_patterns = [
                                r"```\s*yaml\s*(.*?)```",
                                r"```\s*YAML\s*(.*?)```",
                                r"```(.*?)```",
                                r"`{3,}(.*?)`{3,}",
                            ]

                            for pattern in yaml_patterns:
                                match = re.search(pattern, text, re.DOTALL)
                                if match:
                                    extracted = match.group(1).strip()
                                    # Basic validation: extracted content should have some structure
                                    if ":" in extracted and len(extracted) > 10:
                                        return extracted

                            # If we got here, try to find any YAML-like content
                            # Look for beginning of what appears to be YAML content
                            if (
                                "test_name:" in text
                                or "user:" in text
                                or "chatbot:" in text
                            ):
                                # Try to extract what looks like YAML even without code fences
                                lines = text.strip().split("\n")
                                # Skip any non-YAML looking lines at the beginning
                                while lines and not any(
                                    keyword in lines[0]
                                    for keyword in [
                                        "test_name:",
                                        "user:",
                                        "chatbot:",
                                        "llm:",
                                    ]
                                ):
                                    lines.pop(0)
                                return "\n".join(lines)

                            # Last resort: return the whole response stripped
                            return text.strip()

                        try:
                            just_yaml = _extract_yaml(fixed_yaml_str)
                            fixed_profile = yaml.safe_load(just_yaml)
                            re_errors = validator.validate(just_yaml)

                            if not re_errors:
                                print("  ✓ Profile fixed successfully!")
                                validated_profiles.append(fixed_profile)
                            else:
                                print(
                                    f"  ✗ LLM couldn't fix all errors ({len(re_errors)} remaining)"
                                )
                                validated_profiles.append(profile)
                        except Exception as e:
                            print(f"  ✗ Failed to parse LLM's YAML: {type(e).__name__}")
                            validated_profiles.append(profile)
                    except Exception as e:
                        print(f"  ✗ LLM call failed: {type(e).__name__}")
                        validated_profiles.append(profile)

            return {
                "messages": state["messages"],
                "built_profiles": validated_profiles,
            }
        return {"messages": state["messages"]}

    def _structure_builder_node(self, state: State) -> State:
        """
        Analyzes the flat list of functionalities and conversation history
        to build a structured workflow graph (represented as nested dicts).
        """
        if not state.get("exploration_finished", False):
            print("Skipping structure builder: Exploration not finished.")
            return state

        print("\n--- Building Workflow Structure ---")
        flat_functionality_dicts = state.get("discovered_functionalities", [])
        conversation_history = state.get("conversation_history", [])

        if not flat_functionality_dicts:
            print("   Skipping structure building: No initial functionalities found.")
            # Keep functionalities as an empty list
            return {**state, "discovered_functionalities": []}

        # Prepare input for the structuring LLM call
        func_list_str = "\n".join(
            [
                f"- Name: {f.get('name', 'N/A')}\n  Description: {f.get('description', 'N/A')}\n  Parameters: {', '.join(p.get('name', '?') for p in f.get('parameters', [])) or 'None'}"
                for f in flat_functionality_dicts
            ]
        )

        structuring_prompt = f"""
        You are a Workflow Dependency Analyzer specializing in sequential process modeling. Your task is to analyze a list of discovered chatbot functionalities (actions/steps) and conversation transcripts to determine the EXACT sequential workflow a user must follow.

        Input Functionalities:
        {func_list_str}

        Conversation History Snippets:
        {str(conversation_history)[:3000]} # Include a portion of the history for context

        CRITICAL TASK: Determine which functionalities depend on others and CANNOT be accessed without first completing their prerequisites.

        EXAMPLE:
        In an ordering system:
        - "order_drinks" can ONLY happen AFTER "order_pizza" because the workflow forces users to order main items first
        - "confirm_order" can ONLY happen AFTER all items are selected

        When analyzing the conversation flow, identify:
        1. Mandatory starting points in the workflow
        2. Actions that are only available after completing prerequisite steps
        3. The actual ORDER in which users must perform actions

        DEEPLY ANALYZE:
        1. Which actions must be performed FIRST before others become available?
        2. Which actions are sub-steps of larger processes?
        3. Where in conversations do users get REDIRECTED to complete prerequisite steps?

        Structure the output as a JSON list of nodes, where each node represents a functionality and includes its parent(s).

        Rules:
        - A node with parents can ONLY be accessed AFTER its parent nodes are completed
        - Root nodes (no parents) are the ONLY valid starting points in the conversation
        - Use the 'name' field from the input functionalities as the primary identifier
        - The output MUST be valid JSON - DO NOT include comments
        - Empty arrays should be represented as [] without comments

        Output Format Example:
        [
        {{
            "name": "start_order",
            "description": "Begin a new order",
            "parameters": [],
            "parent_names": []
        }},
        {{
            "name": "select_main_item",
            "description": "Select a main item for the order",
            "parameters": ["item_type"],
            "parent_names": ["start_order"]
        }},
        {{
            "name": "add_side_items",
            "description": "Add side items to the order",
            "parameters": ["side_item_type"],
            "parent_names": ["select_main_item"]
        }},
        {{
            "name": "checkout",
            "description": "Complete the order",
            "parameters": ["payment_method"],
            "parent_names": ["select_main_item"]
        }}
        ]

        PAY CLOSE ATTENTION to the conversation flow to identify true dependencies, not just related concepts. Actions that can only happen AFTER another action MUST list that action in parent_names.

        Generate the JSON list representing the precise sequential workflow structure:
        """

        try:
            print("   Asking LLM to determine workflow structure...")
            response = self.llm.invoke(structuring_prompt)
            response_content = response.content

            # --- Extract JSON from LLM response ---
            # Try multiple patterns to extract JSON
            json_str = None
            json_patterns = [
                r"```json\s*([\s\S]+?)\s*```",  # JSON with code fence
                r"```\s*([\s\S]+?)\s*```",  # Any code fence
                r"\[\s*\{.+\}\s*\]",  # Raw JSON array pattern
            ]

            for pattern in json_patterns:
                match = re.search(pattern, response_content, re.DOTALL)
                if match:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    break

            if not json_str:
                # Last resort - assume the entire content might be JSON
                json_str = response_content.strip()

            print("   Parsing workflow structure from LLM response...")
            # Remove JavaScript-style comments before parsing
            json_str = re.sub(r"//.*?(\n|$)", "\n", json_str)

            # Handle cleanup of any trailing commas which might be in the JSON
            json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)

            structured_nodes_info = json.loads(json_str)
            if not isinstance(structured_nodes_info, list):
                raise ValueError("LLM response is not a JSON list.")

            # --- Rebuild Node Dicts with Hierarchy ---
            # Create a map of nodes by name for easy lookup
            nodes_map: Dict[str, Dict[str, Any]] = {
                node_info["name"]: node_info
                for node_info in structured_nodes_info
                if "name" in node_info
            }
            # Clear existing children and add based on parent_names
            for node_info in nodes_map.values():
                node_info["children"] = []  # Reset children list

            # Add children based on parent_names
            for node_name, node_info in nodes_map.items():
                parent_names = node_info.get("parent_names", [])
                if not parent_names:
                    # If no parents, it's potentially a root node (will verify later)
                    pass  # Handled in the next loop
                else:
                    for parent_name in parent_names:
                        if parent_name in nodes_map:
                            parent_node_info = nodes_map[parent_name]
                            # Add current node's dict as a child to parent's dict
                            # Avoid adding duplicates if structure is complex
                            if node_info not in parent_node_info.get("children", []):
                                parent_node_info.setdefault("children", []).append(
                                    node_info
                                )
                        else:
                            print(
                                f"   Warning: Parent '{parent_name}' listed for node '{node_name}' not found in nodes map."
                            )

            # Identify true root nodes (those not listed as children anywhere)
            all_child_names = set()
            for node_info in nodes_map.values():
                for child_info in node_info.get("children", []):
                    all_child_names.add(child_info["name"])

            root_nodes_dicts = [
                node_info
                for node_name, node_info in nodes_map.items()
                if node_name not in all_child_names
            ]

            print(f"   Built structure with {len(root_nodes_dicts)} root node(s).")

            # Check for improper nesting - identify functionalities that should be nested
            # but aren't properly connected in the parent-child relationships
            print("   Verifying proper nesting of dependent functionalities...")
            all_node_names = set(nodes_map.keys())
            potentially_misplaced = []

            # Look for nodes with names suggesting they should be nested
            for node_name in all_node_names:
                if "add" in node_name.lower() or "order_drink" in node_name.lower():
                    node = nodes_map[node_name]
                    # Check if this should likely be nested under a parent
                    if (
                        not node.get("parent_names")
                        and node_name not in all_child_names
                    ):
                        potentially_misplaced.append(node_name)

            if potentially_misplaced:
                print(
                    f"   Found {len(potentially_misplaced)} potentially misplaced nodes: {', '.join(potentially_misplaced)}"
                )
                print("   Attempting to correct workflow hierarchy...")

                # Ask the LLM to specifically review these nodes
                correction_prompt = f"""
                Review these potentially misplaced nodes in our workflow: {", ".join(potentially_misplaced)}

                These nodes might need parents but were identified as root nodes. Review them carefully.

                Current workflow structure:
                {json.dumps(root_nodes_dicts, indent=2)[:1000]}

                For each potentially misplaced node, determine:
                1. Should it be a child of another node? If yes, which one(s)?
                2. Or is it correctly a root/starting node?

                Return ONLY a JSON list of corrections with this structure:
                [
                    {{
                        "node_name": "name_of_node",
                        "should_be_child_of": ["parent1", "parent2"]
                    }},
                    ...
                ]
                """

                try:
                    correction_response = self.llm.invoke(correction_prompt)
                    correction_content = correction_response.content

                    # Extract JSON from correction response
                    match = re.search(
                        r"\[\s*\{.+\}\s*\]", correction_content, re.DOTALL
                    )
                    if match:
                        corrections = json.loads(match.group(0))

                        # Apply corrections
                        for correction in corrections:
                            node_name = correction.get("node_name")
                            parents = correction.get("should_be_child_of", [])

                            if node_name in nodes_map and parents:
                                print(
                                    f"   Correcting: '{node_name}' should be child of {parents}"
                                )

                                # Update the node's parent_names
                                nodes_map[node_name]["parent_names"] = parents

                                # Re-establish parent-child relationships
                                for parent_name in parents:
                                    if parent_name in nodes_map:
                                        # Remove from root nodes if it was there
                                        if node_name in [
                                            n.get("name") for n in root_nodes_dicts
                                        ]:
                                            root_nodes_dicts = [
                                                n
                                                for n in root_nodes_dicts
                                                if n.get("name") != node_name
                                            ]

                                        # Add as child to parent
                                        parent_node = nodes_map[parent_name]
                                        if nodes_map[node_name] not in parent_node.get(
                                            "children", []
                                        ):
                                            parent_node.setdefault(
                                                "children", []
                                            ).append(nodes_map[node_name])

                        # Recalculate root nodes after corrections
                        all_child_names = set()
                        for node_info in nodes_map.values():
                            for child_info in node_info.get("children", []):
                                all_child_names.add(child_info["name"])

                        root_nodes_dicts = [
                            node_info
                            for node_name, node_info in nodes_map.items()
                            if node_name not in all_child_names
                        ]

                        print(
                            f"   After corrections: {len(root_nodes_dicts)} root node(s)"
                        )

                except Exception as correction_error:
                    print(f"   Error applying corrections: {correction_error}")

            # Update state with the structured list of dictionaries
            return {**state, "discovered_functionalities": root_nodes_dicts}

        except json.JSONDecodeError as e:
            print(f"   Error: Failed to decode JSON from LLM response: {e}")
            print(f"   LLM Response Content:\n{response_content}")
            # Keep the flat list if structuring fails
            return state
        except Exception as e:
            print(f"   Error during structure building: {e}")
            # Keep the flat list if structuring fails
            return state
