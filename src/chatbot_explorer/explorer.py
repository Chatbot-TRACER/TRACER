from typing import Annotated, Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import os
import re
import yaml

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
        graph_builder.add_node("goal_generator", self._goal_generator_node)
        graph_builder.add_node("conversation_params", self._conversation_params_node)
        graph_builder.add_node("profile_builder", self._build_profiles_node)
        graph_builder.add_node("profile_validator", self._validate_profiles_node)

        # Add edges
        graph_builder.set_entry_point("explorer")
        graph_builder.add_edge("explorer", "analyzer")
        graph_builder.add_edge("analyzer", "goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        return graph_builder.compile(checkpointer=self.memory)

    def _explorer_node(self, state: State):
        """Explorer node for interacting with the target chatbot."""
        if not state["exploration_finished"]:
            return {"messages": [self.llm.invoke(state["messages"])], "explored": True}
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
            print("\n--- Generating conversation goals ---")

            # Convert list of dicts into descriptions
            functionality_dicts: List[Dict[str, Any]] = state[
                "discovered_functionalities"
            ]
            functionality_descriptions: List[str] = [
                func["description"]
                for func in functionality_dicts
                if "description" in func
            ]

            if not functionality_descriptions:
                print(
                    "   Warning: No descriptions found in functionalities to generate goals."
                )
                return state

            print(
                f" -> Preparing {len(functionality_descriptions)} descriptions for goal generation prompt."
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

            profiles_with_params = generate_conversation_parameters(
                state["conversation_goals"],
                state["discovered_functionalities"],
                self.llm,
                supported_languages=state["supported_languages"],
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
