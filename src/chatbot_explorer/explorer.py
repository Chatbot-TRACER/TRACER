from typing import Annotated, Dict, List, Any, Optional, Set
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import os
import re
import yaml
import json
import random

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from .nodes.goals_node import generate_user_profiles_and_goals
from .nodes.analyzer_node import analyze_conversations
from .nodes.conversation_parameters_node import generate_conversation_parameters

from .validation_script import YamlValidator

from .functionality_node import FunctionalityNode
from .session import (
    format_conversation,
    run_exploration_session,
    extract_supported_languages,
    extract_fallback_message,
)

# Regex to find {{variables}}
VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")

# Personalities to choose one for the profile
AVAILABLE_PERSONALITIES = [
    "conversational-user",
    "curious-user",
    "direct-user",
    "disorganized-user",
    "elderly-user",
    "formal-user",
    "impatient-user",
    "rude-user",
    "sarcastic-user",
    "skeptical-user",
]


class State(TypedDict):
    """Holds the state for the graph."""

    messages: Annotated[list, add_messages]  # Chat messages
    conversation_history: list  # History of all sessions
    discovered_functionalities: List[FunctionalityNode]  # Found features
    discovered_limitations: list  # Found limits
    current_session: int  # Which session number we're on
    exploration_finished: bool  # Flag if exploration is done
    conversation_goals: list  # Goals for generating profiles
    supported_languages: list  # Languages the bot speaks
    built_profiles: list  # Generated YAML profiles
    fallback_message: str  # Bot's fallback message
    chatbot_type: str  # Type of chatbot: "transactional", "informational", or "unknown"


class ChatbotExplorer:
    """Uses LangGraph to explore chatbots."""

    def __init__(self, model_name: str):
        """
        Sets up the explorer.

        Args:
            model_name (str): The name of the OpenAI model to use.
        """
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()  # For saving graph state
        self.graph = self._build_graph()  # Build the main graph

    def run_full_exploration(
        self, chatbot_connector, max_sessions: int, max_turns: int
    ) -> Dict[str, Any]:
        """
        Runs the initial probing and the main exploration loop.

        Args:
            chatbot_connector: An instance of a chatbot connector class.
            max_sessions (int): Maximum number of exploration sessions to run.
            max_turns (int): Maximum turns per exploration session.

        Returns:
            Dict[str, Any]: A dictionary containing exploration results:
                            - conversation_sessions: List of conversation histories.
                            - root_nodes_dict: List of root FunctionalityNode objects as dicts.
                            - supported_languages: List of detected languages.
                            - fallback_message: Detected fallback message string.
        """
        # Initialize results and state tracking
        conversation_sessions: List[List[Dict[str, str]]] = []
        supported_languages: List[str] = []
        fallback_message: Optional[str] = None
        root_nodes: List[FunctionalityNode] = []
        pending_nodes: List[FunctionalityNode] = []
        explored_nodes: Set[str] = set()

        # --- Initial Language Detection ---
        print("\n--- Probing Chatbot Language ---")
        initial_probe_query = "Hello"  # Simple initial query
        is_ok, probe_response = chatbot_connector.execute_with_input(
            initial_probe_query
        )
        if is_ok and probe_response:
            print(f"   Initial response received: '{probe_response[:60]}...'")
            try:
                # Attempt language detection via LLM
                detected_langs = extract_supported_languages(probe_response, self.llm)
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

        # --- Initial Fallback Message Detection ---
        print("\n--- Attempting to detect chatbot fallback message ---")
        try:
            # Call the function using self.llm and the connector
            fallback_message = extract_fallback_message(chatbot_connector, self.llm)
            if fallback_message:
                print(f'   Detected fallback message: "{fallback_message[:50]}..."')
            else:
                print("   Could not detect a fallback message.")
        except Exception as fb_e:
            print(
                f"   Error during fallback detection: {fb_e}. Proceeding without fallback."
            )
            fallback_message = None
        # --- End Fallback Message Detection ---

        # --- Exploration Loop ---
        session_num = 0
        while session_num < max_sessions:
            current_session_index = session_num
            print(
                f"\n=== Starting Session {current_session_index + 1}/{max_sessions} ==="
            )

            explore_node = None  # Node to focus on this session
            session_type_log = "General Exploration"

            if pending_nodes:
                # Prioritize exploring specific nodes from the queue
                explore_node = pending_nodes.pop(0)
                # Double-check if node was already explored
                if explore_node.name in explored_nodes:
                    print(
                        f"--- Skipping already explored node: '{explore_node.name}' ---"
                    )
                    session_num += 1  # Consume a session slot
                    continue
                session_type_log = f"Exploring functionality '{explore_node.name}'"
            elif (
                session_num > 0
            ):  # If queue is empty after session 0, perform general exploration
                print(
                    "   Pending nodes queue is empty. Performing general exploration."
                )
            # Else: Session 0 and queue is empty is the initial state.

            print(f"   Session Type: {session_type_log}")

            # Execute one exploration session
            (
                conversation_history,
                _,  # Languages (we already detected initial ones)
                _,  # New nodes (handled within run_exploration_session)
                updated_roots,  # Updated list of root nodes
                updated_pending,  # Updated queue of nodes to explore
                updated_explored,  # Updated set of explored node names
            ) = run_exploration_session(
                current_session_index,  # Pass 0-based index
                max_sessions,  # Use max_sessions parameter
                max_turns,  # Use max_turns parameter
                self,  # Pass self (the explorer instance)
                chatbot_connector,  # Pass the connector instance
                fallback_message=fallback_message,
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

            # Move to the next session
            session_num += 1
        # --- End Exploration Loop ---

        # --- Post-Exploration Summary ---
        if session_num == max_sessions:  # Use max_sessions parameter
            print(
                f"\n=== Completed {max_sessions} exploration sessions ==="
            )  # Use max_sessions
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

        # Convert FunctionalityNode objects to dictionaries for the return value
        functionality_dicts = [node.to_dict() for node in root_nodes]

        # Return the collected results
        return {
            "conversation_sessions": conversation_sessions,
            "root_nodes_dict": functionality_dicts,  # Use the converted dicts
            "supported_languages": supported_languages,
            "fallback_message": fallback_message,
        }

    def _build_graph(self):
        """
        Builds the main LangGraph for exploration and analysis.

        Returns:
            CompiledGraph: The compiled LangGraph application.
        """
        graph_builder = StateGraph(State)

        # Add graph nodes
        graph_builder.add_node("explorer", self._explorer_node)
        graph_builder.add_node("analyzer", self._analyzer_node)
        graph_builder.add_node("structure_builder", self._structure_builder_node)
        graph_builder.add_node("goal_generator", self._goal_generator_node)
        graph_builder.add_node("conversation_params", self._conversation_params_node)
        graph_builder.add_node("profile_builder", self._build_profiles_node)
        graph_builder.add_node("profile_validator", self._validate_profiles_node)

        # Define the flow (edges)
        graph_builder.set_entry_point("explorer")
        graph_builder.add_edge("explorer", "analyzer")
        graph_builder.add_edge("analyzer", "structure_builder")
        graph_builder.add_edge("structure_builder", "goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        # Compile the graph
        return graph_builder.compile(checkpointer=self.memory)

    def _build_profile_generation_graph(self):
        """
        Builds a smaller graph just for generating profiles from existing structure.

        Returns:
            CompiledGraph: The compiled LangGraph application for profile generation.
        """
        graph_builder = StateGraph(State)

        # Add nodes needed for profile generation
        graph_builder.add_node("goal_generator", self._goal_generator_node)
        graph_builder.add_node("conversation_params", self._conversation_params_node)
        graph_builder.add_node("profile_builder", self._build_profiles_node)
        graph_builder.add_node("profile_validator", self._validate_profiles_node)

        # Define the flow
        graph_builder.set_entry_point("goal_generator")
        graph_builder.add_edge("goal_generator", "conversation_params")
        graph_builder.add_edge("conversation_params", "profile_builder")
        graph_builder.add_edge("profile_builder", "profile_validator")
        graph_builder.set_finish_point("profile_validator")

        # Compile the graph
        return graph_builder.compile(checkpointer=self.memory)

    def _build_structure_graph(self):
        """
        Builds a graph that only runs the structure building part.

        Returns:
            CompiledGraph: The compiled LangGraph application for structure building.
        """
        graph_builder = StateGraph(State)

        # Add only the structure builder node
        graph_builder.add_node("structure_builder", self._structure_builder_node)

        # Start and end with this node
        graph_builder.set_entry_point("structure_builder")
        graph_builder.set_finish_point("structure_builder")

        # Compile the graph
        return graph_builder.compile(checkpointer=self.memory)

    def _explorer_node(self, state: State):
        """
        Node that handles the actual chat interaction (if exploration isn't finished).

        Args:
            state (State): The current graph state.

        Returns:
            dict: Updated state dictionary with new messages.
        """
        if not state["exploration_finished"]:
            max_history_messages = 20  # Limit context window (last 10 turns)
            messages_to_send = []
            if state["messages"]:
                messages_to_send.append(
                    state["messages"][0]
                )  # Always keep system prompt
                messages_to_send.extend(
                    state["messages"][-max_history_messages:]
                )  # Add recent messages

            if not messages_to_send:  # Safety check for empty state
                return {"messages": state["messages"]}

            # Call the LLM to get the next message
            return {"messages": [self.llm.invoke(messages_to_send)], "explored": True}

        # If exploration is finished, just pass the state along
        return {"messages": state["messages"]}

    def _analyzer_node(self, state: State) -> State:
        """
        Node that analyzes conversation history to find functionalities and limitations.

        Args:
            state (State): The current graph state.

        Returns:
            State: Updated state with discovered functionalities and limitations.
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
            # Call the analysis function
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

        # Convert FunctionalityNode objects to simple dictionaries for state
        functionality_dicts = [
            func.to_dict() for func in analysis_output.get("functionalities", [])
        ]

        # Update the state
        output_state = {**state}
        output_state["messages"] = state.get("messages", []) + [
            analysis_output.get("analysis_result", "Analysis log missing.")
        ]
        # Store the dictionaries
        output_state["discovered_functionalities"] = functionality_dicts
        output_state["discovered_limitations"] = analysis_output.get("limitations", [])

        return output_state

    def _goal_generator_node(self, state: State) -> State:
        """
        Node that generates user profiles and conversation goals based on findings.

        Args:
            state (State): The current graph state.

        Returns:
            State: Updated state with conversation goals.
        """
        if state.get("exploration_finished", False) and state.get(
            "discovered_functionalities"
        ):
            print("\n--- Generating conversation goals from structured data ---")

            # Functionalities are now dicts
            structured_root_dicts: List[Dict[str, Any]] = state[
                "discovered_functionalities"
            ]

            # Get workflow structure from state if available
            workflow_structure = state.get("workflow_structure", None)

            # Get chatbot type from state if available
            chatbot_type = state.get("chatbot_type", "unknown")
            print(f"   Chatbot type for goal generation: {chatbot_type}")

            # Helper to get all descriptions from the structure
            def get_all_descriptions(nodes: List[Dict[str, Any]]) -> List[str]:
                descriptions = []
                for node in nodes:
                    if "description" in node and node["description"]:
                        descriptions.append(node["description"])
                    if "children" in node and node["children"]:
                        child_descriptions = get_all_descriptions(node["children"])
                        descriptions.extend(child_descriptions)
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
                # Call the goal generation function with workflow structure and chatbot type
                profiles_with_goals = generate_user_profiles_and_goals(
                    functionality_descriptions,
                    state.get("discovered_limitations", []),
                    self.llm,
                    workflow_structure=workflow_structure,
                    conversation_history=state.get("conversation_history", []),
                    supported_languages=state.get("supported_languages", []),
                    chatbot_type=chatbot_type,
                )
                print(f" -> Generated {len(profiles_with_goals)} profiles with goals.")
                # Update state with goals
                return {**state, "conversation_goals": profiles_with_goals}

            except Exception as e:
                print(f"Error during goal generation: {e}")
                return {**state, "conversation_goals": []}  # Return empty list on error

        elif state.get("exploration_finished", False):
            print("\n--- Skipping goal generation: No functionalities discovered. ---")
            return state
        else:
            print("\n--- Skipping goal generation: Exploration not finished. ---")
            return state

    def _conversation_params_node(self, state: State):
        """
        Node that generates specific parameters needed for conversation goals.

        Args:
            state (State): The current graph state.

        Returns:
            dict: Updated state dictionary with parameters added to goals.
        """
        if state["exploration_finished"] and state["conversation_goals"]:
            print("\n--- Generating conversation parameters ---")

            # Functionalities are dicts
            structured_root_dicts = state.get("discovered_functionalities", [])

            # Helper to flatten the structure info
            def get_all_func_info(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                all_info = []
                for node in nodes:
                    # Get node info (excluding children for flat list)
                    info = {k: v for k, v in node.items() if k != "children"}
                    all_info.append(info)
                    if node.get("children"):  # Recursively add children info
                        all_info.extend(get_all_func_info(node["children"]))
                return all_info

            flat_func_info = get_all_func_info(structured_root_dicts)

            # Call parameter generation function
            profiles_with_params = generate_conversation_parameters(
                state["conversation_goals"],
                flat_func_info,  # Pass the flat list
                self.llm,
                supported_languages=state.get("supported_languages", []),
            )

            # Update state (only need to update goals)
            return {
                "messages": state["messages"],
                "conversation_goals": profiles_with_params,
            }
        # Return unchanged state if skipped
        return {"messages": state["messages"]}

    def _classify_chatbot_type(self, state: State) -> str:
        """
        Tries to guess if the chatbot is more for tasks (transactional) or answering questions (informational).

        Args:
            state (State): The current graph state.

        Returns:
            str: "transactional", "informational", or "unknown".
        """
        print("\n--- Classifying Chatbot Interaction Type ---")
        flat_functionality_dicts = state.get("discovered_functionalities", [])
        conversation_history = state.get("conversation_history", [])

        if not conversation_history or not flat_functionality_dicts:
            print("   Skipping classification: Insufficient data.")
            return "unknown"

        # Summarize functionalities
        func_summary = "\n".join(
            [
                f"- {f.get('name')}: {f.get('description')[:100]}..."
                for f in flat_functionality_dicts[:10]  # Limit summary size
            ]
        )

        # Get snippets from conversation history
        snippets = []
        total_snippet_length = 0
        max_total_snippet_length = 5000  # Limit context size

        if isinstance(conversation_history, list):
            for i, session_history in enumerate(conversation_history):
                if not isinstance(session_history, list):
                    continue
                session_str = format_conversation(session_history)  # Format session
                snippet_len = 1000  # Max length per snippet
                # Take beginning and end if too long
                session_snippet = (
                    session_str[: snippet_len // 2]
                    + "\n...\n"
                    + session_str[-snippet_len // 2 :]
                    if len(session_str) > snippet_len
                    else session_str
                )
                # Add snippet if within total length limit
                if (
                    total_snippet_length + len(session_snippet)
                    < max_total_snippet_length
                ):
                    snippets.append(
                        f"\n--- Snippet from Session {i + 1} ---\n{session_snippet}"
                    )
                    total_snippet_length += len(session_snippet)
                else:
                    break  # Stop if limit reached
        conversation_snippets = "\n".join(snippets)
        if not conversation_snippets:
            conversation_snippets = "No conversation snippets available."

        # Prompt for classification
        classification_prompt = f"""
        Analyze the following conversation snippets and discovered functionalities to classify the chatbot's primary interaction style.

        Discovered Functionalities Summary:
        {func_summary}

        Conversation Snippets:
        {conversation_snippets}

        Consider these definitions:
        - **Transactional / Workflow-driven:** The chatbot guides the user through a specific multi-step process with clear sequences, choices, and goals (e.g., ordering food, booking an appointment, completing a form). Conversations often involve the chatbot asking questions to gather input and presenting options to advance the workflow.
        - **Informational / Q&A:** The chatbot primarily answers user questions on various independent topics. Users typically ask a question, get an answer (often text or links), and might then ask about a completely different topic. There isn't usually a strict required sequence between topics.

        Based on the overall pattern in the conversations and the nature of the functionalities, is this chatbot PRIMARILY Transactional/Workflow-driven or Informational/Q&A?

        Respond with ONLY ONE word: "transactional" or "informational".
        """

        try:
            # Ask the LLM
            response = self.llm.invoke(classification_prompt)
            classification = response.content.strip().lower()
            if classification in ["transactional", "informational"]:
                print(f"   LLM classified as: {classification}")
                return classification
            else:
                # Handle unclear response
                print(
                    f"   LLM response unclear ('{classification}'), defaulting to informational."
                )
                return "informational"
        except Exception as e:
            # Handle LLM error
            print(f"   Error during classification: {e}. Defaulting to informational.")
            return "informational"

    def _build_profiles_node(self, state: State):
        """
        Node that takes conversation goals and builds the final YAML profile dictionaries.

        Args:
            state (State): The current graph state.

        Returns:
            dict: Updated state dictionary with 'built_profiles'.
        """
        if state["exploration_finished"] and state["conversation_goals"]:
            print("\n--- Building user profiles ---")
            built_profiles = []

            # Get fallback message (or use a default)
            fallback_message = state.get(
                "fallback_message", "I'm sorry, I don't understand."
            )

            # Get primary language (or default to English)
            primary_language = "English"
            if (
                state.get("supported_languages")
                and len(state["supported_languages"]) > 0
            ):
                primary_language = state["supported_languages"][0]

            # Build YAML for each profile goal set
            for profile in state["conversation_goals"]:
                profile_yaml = self._build_profile_yaml(
                    profile,
                    fallback_message=fallback_message,
                    primary_language=primary_language,
                )
                built_profiles.append(profile_yaml)
            # Update state with the list of profile dicts
            return {"messages": state["messages"], "built_profiles": built_profiles}
        # Return unchanged state if skipped
        return {"messages": state["messages"]}

    def run_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Runs the full graph once.

        Args:
            state (dict): The initial state for the graph.
            config (dict, optional): Configuration for the graph run (like thread_id). Defaults to None.

        Returns:
            dict: The final state after the graph run.
        """
        if config is None:
            config = {"configurable": {"thread_id": "1"}}  # Default config
        return self.graph.invoke(state, config=config)

    def stream_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Streams the graph execution step by step.

        Args:
            state (dict): The initial state for the graph.
            config (dict, optional): Configuration for the graph run. Defaults to None.

        Returns:
            Iterator: An iterator yielding state updates at each step.
        """
        if config is None:
            config = {"configurable": {"thread_id": "1"}}  # Default config
        return self.graph.stream(state, config=config)

    def _build_profile_yaml(self, profile, fallback_message, primary_language):
        """
        Helper function to create the dictionary structure for a single YAML profile.

        Args:
            profile (dict): The profile data including goals and parameters.
            fallback_message (str): The chatbot's fallback message.
            primary_language (str): The primary language for the user.

        Returns:
            dict: A dictionary representing the YAML profile structure.
        """
        # Find all {{variables}} used *only* in the string goals
        used_variables = set()
        original_goals_list = profile.get("goals", [])  # Get the mixed list

        for goal_item in original_goals_list:
            if isinstance(goal_item, str):
                # Only apply findall to strings
                variables_in_string_goal = VARIABLE_PATTERN.findall(goal_item)
                used_variables.update(variables_in_string_goal)

        # Create the goals list for YAML (mix of strings and variable dicts)
        yaml_goals = list(profile.get("goals", []))
        for var_name in used_variables:
            if var_name in profile:  # Add variable definition if found in profile data
                yaml_goals.append({var_name: profile[var_name]})

        # Build the chatbot section
        chatbot_section = {
            "is_starter": False,  # Assuming chatbot doesn't start
            "fallback": fallback_message,
        }
        if "outputs" in profile:  # Add expected outputs if any
            chatbot_section["output"] = profile["outputs"]

        # Build the user context list
        user_context = []

        # 75% with personalit
        if random.random() < 0.75:
            selected_personality = random.choice(AVAILABLE_PERSONALITIES)
            user_context.append(f"personality: personalities/{selected_personality}")

        # Choose a random temperature
        temperature = round(random.uniform(0.3, 1.0), 1)

        context = profile.get("context", [])
        # Add other context items
        if isinstance(context, str):
            user_context.append(context)
        else:
            user_context.extend(context)

        # Get conversation settings
        conversation_section = profile.get("conversation", {})

        # Assemble the final profile dictionary
        return {
            "test_name": profile["name"],
            "llm": {
                "temperature": temperature,
                "model": "gpt-4o-mini",
                "format": {"type": "text"},
            },
            "user": {
                "language": primary_language,
                "role": profile["role"],
                "context": user_context,
                "goals": yaml_goals,
            },
            "chatbot": chatbot_section,
            "conversation": conversation_section,
        }

    def _validate_profiles_node(self, state: State):
        """
        Node that validates generated YAML profiles and tries to fix them using LLM if needed.

        Args:
            state (State): The current graph state.

        Returns:
            dict: Updated state dictionary with validated (and potentially fixed) 'built_profiles'.
        """
        if state["exploration_finished"] and state.get("built_profiles"):
            print("\n--- Validating user profiles ---")
            validator = YamlValidator()  # Our validator class
            validated_profiles = []  # List to hold good profiles

            for profile in state["built_profiles"]:
                # Convert profile dict back to YAML string for validation
                yaml_content = yaml.dump(profile, sort_keys=False, allow_unicode=True)
                errors = validator.validate(yaml_content)  # Validate

                if not errors:
                    # Profile is valid
                    validated_profiles.append(profile)
                    print(f"✓ Profile '{profile['test_name']}' valid, no fixes needed.")
                else:
                    # Profile has errors
                    error_count = len(errors)
                    print(
                        f"\n⚠ Profile '{profile['test_name']}' has {error_count} validation errors"
                    )

                    # Print first few errors
                    for e in errors[:3]:
                        print(f"  • {e.path}: {e.message}")
                    if error_count > 3:
                        print(f"  • ... and {error_count - 3} more errors")

                    # Prepare prompt for LLM to fix errors
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
                        # Ask LLM to fix it
                        fixed_yaml_str = self.llm.invoke(
                            [{"role": "user", "content": fix_prompt}]
                        )

                        # Helper to extract YAML from LLM response (handles ```yaml ... ``` etc.)
                        def _extract_yaml(text: str) -> str:
                            if hasattr(
                                text, "content"
                            ):  # Handle LangChain message object
                                text = text.content

                            # Try common code fence patterns
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
                                    # Basic check if it looks like YAML
                                    if ":" in extracted and len(extracted) > 10:
                                        return extracted

                            # If no fences, check if it starts like YAML
                            if (
                                "test_name:" in text
                                or "user:" in text
                                or "chatbot:" in text
                            ):
                                # Try to strip leading non-YAML lines
                                lines = text.strip().split("\n")
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

                            # Give up and return stripped text
                            return text.strip()

                        try:
                            # Extract and parse the fixed YAML
                            just_yaml = _extract_yaml(fixed_yaml_str)
                            fixed_profile = yaml.safe_load(just_yaml)
                            # Re-validate the fixed YAML
                            re_errors = validator.validate(just_yaml)

                            if not re_errors:
                                # Fixed successfully!
                                print("  ✓ Profile fixed successfully!")
                                validated_profiles.append(fixed_profile)
                            else:
                                # Still has errors, keep original
                                print(
                                    f"  ✗ LLM couldn't fix all errors ({len(re_errors)} remaining)"
                                )
                                validated_profiles.append(profile)  # Keep original
                        except Exception as e:
                            # Failed to parse LLM output, keep original
                            print(f"  ✗ Failed to parse LLM's YAML: {type(e).__name__}")
                            validated_profiles.append(profile)  # Keep original
                    except Exception as e:
                        # LLM call failed, keep original
                        print(f"  ✗ LLM call failed: {type(e).__name__}")
                        validated_profiles.append(profile)  # Keep original

            # Update state with the list of validated (or original if fix failed) profiles
            return {
                "messages": state["messages"],
                "built_profiles": validated_profiles,
            }
        # Return unchanged state if skipped
        return {"messages": state["messages"]}

    def _structure_builder_node(self, state: State) -> State:
        """
        Node that analyzes functionalities and history to build the workflow structure.
        Uses different logic based on whether the bot seems transactional or informational.

        Args:
            state (State): The current graph state.

        Returns:
            State: Updated state with structured 'discovered_functionalities'.
        """
        if not state.get("exploration_finished", False):
            print("Skipping structure builder: Exploration not finished.")
            return state

        print("\n--- Building Workflow Structure ---")
        flat_functionality_dicts = state.get("discovered_functionalities", [])
        conversation_history = state.get("conversation_history", [])

        if not flat_functionality_dicts:
            print("   Skipping structure building: No initial functionalities found.")
            return {
                **state,
                "discovered_functionalities": [],
            }  # Ensure it's an empty list

        # Classify the bot type first
        bot_type = self._classify_chatbot_type(state)

        # Store the bot type in the state
        state = {**state, "chatbot_type": bot_type}

        # Prepare input for the LLM (functionality list and conversation snippets)
        func_list_str = "\n".join(
            [
                f"- Name: {f.get('name', 'N/A')}\n  Description: {f.get('description', 'N/A')}\n  Parameters: {', '.join(p.get('name', '?') for p in f.get('parameters', [])) or 'None'}"
                for f in flat_functionality_dicts
            ]
        )

        # Get conversation snippets (same logic as classification)
        snippets = []
        total_snippet_length = 0
        max_total_snippet_length = 7000  # Larger context for structure

        if isinstance(conversation_history, list):
            for i, session_history in enumerate(conversation_history):
                if not isinstance(session_history, list):
                    continue
                session_str = format_conversation(session_history)
                snippet_len = 1500
                session_snippet = (
                    session_str[: snippet_len // 2]
                    + "\n...\n"
                    + session_str[-snippet_len // 2 :]
                    if len(session_str) > snippet_len
                    else session_str
                )
                if (
                    total_snippet_length + len(session_snippet)
                    < max_total_snippet_length
                ):
                    snippets.append(
                        f"\n--- Snippet from Session {i + 1} ---\n{session_snippet}"
                    )
                    total_snippet_length += len(session_snippet)
                else:
                    break
        conversation_snippets = "\n".join(snippets)
        if not conversation_snippets:
            conversation_snippets = "No conversation history available."

        # --- Prompts for Transactional vs Informational ---

        # Prompt for Transactional bots (focus on sequence)
        transactional_structuring_prompt = f"""
        You are a Workflow Dependency Analyzer. Analyze the discovered interaction steps (functionalities) and conversation snippets to model the **sequential workflow** a user follows.

        Input Functionalities (Extracted Steps):
        {func_list_str}

        Conversation History Snippets (Context for Flow):
        {conversation_snippets}

        CRITICAL TASK: Determine the sequential flow, including prerequisites, branches, and joins based *primarily on the conversational evidence*. Assume a workflow exists unless proven otherwise.
        - **Sequences:** Identify steps that consistently or logically happen *after* others based on the conversation flow (e.g., selecting size after choosing pizza type).
        - **Branches:** Identify points where the chatbot explicitly offers mutually exclusive choices leading to different subsequent steps (e.g., predefined vs. custom pizza).
        - **Joins:** Identify points where different interaction paths converge to the *same* common next step (e.g., adding drinks after either pizza type).

        **IMPORTANT: Distinguish True Prerequisites from Conversational Sequence:**
        - A step should only have `parent_names` if completing the parent step is **functionally required** to perform the child step. Ask: "Is Step A *necessary* to make Step B possible or meaningful?"
        - **Do NOT assign parentage simply because one step occurred before another in a single conversation.**
        - **Meta-Interactions (like asking "What can you do?", "Help", greetings, asking for general info about the bot itself) should almost always be root nodes (`parent_names: []`)**. They describe the interaction *about* the bot, not the core task flow itself. For example, `inquire_main_functionality` or `ask_capabilities` is NOT a prerequisite for `order_pizza`.

        DEEPLY ANALYZE the conversation flow provided:
        1. Which steps seem like entry points? (Potential root nodes, especially meta-interactions)
        2. Which steps are explicitly offered or occur only *after* another specific step is completed **AND are functionally dependent on it**? (Indicates sequence/parent)
        3. Does the chatbot present clear choices followed by different interactions? (Indicates a branch)
        4. Do different paths seem to lead back to the same follow-up step? (Indicates a join)

        Structure the output as a JSON list of nodes. Each node MUST include:
        - "name": Functionality name (string).
        - "description": Description (string).
        - "parameters": List of parameter names (list of strings or []).
        - "parent_names": List of names of functionalities that, based on conversational evidence AND functional necessity, MUST be completed *immediately before* this one (list of strings). Use `[]` for root nodes and meta-interactions.

        Rules for Output:
        - The structure MUST reflect the likely functional dependencies observed in the conversation flow.
        - Use the 'name' field as the identifier.
        - Output MUST be valid JSON. Use [] for empty lists.

        Generate the JSON list representing the precise sequential workflow structure:
        """

        # Prompt for Informational bots (focus on independence)
        informational_structuring_prompt = f"""
        You are a Workflow Dependency Analyzer. Analyze the discovered interaction steps (functionalities) and conversation snippets to model the interaction flow, recognizing that it might be **non-sequential Q&A**.

        **CRITICAL CONTEXT:** This chatbot appears primarily **Informational/Q&A**. Users likely ask about independent topics.

        Input Functionalities (Extracted Steps):
        {func_list_str}

        Conversation History Snippets (Context from Multiple Sessions):
        {conversation_snippets}

        CRITICAL TASK: Determine relationships based *only* on **strong conversational evidence ACROSS MULTIPLE SESSIONS**.
        - **Sequences/Branches:** Create parent-child relationships (`parent_names`) ONLY IF the chatbot *explicitly forces* a sequence OR if a step is *impossible* without completing a prior one, AND this dependency is observed CONSISTENTLY.
        - **Independent Topics:** If users ask about different topics independently, treat these functionalities as **separate root nodes** (assign `parent_names: []`). **DO NOT infer dependency just because Topic B was discussed after Topic A in one session.**
        - **Meta-Interactions (like asking "What can you do?", "Help", greetings, asking for general info about the bot itself) should almost always be root nodes (`parent_names: []`)**. They describe the interaction *about* the bot, not the core informational topics themselves.

        **RULE: Your DEFAULT action MUST be to create separate root nodes (empty `parent_names`: `[]`). Only create parent-child links if the conversational evidence for functional dependency is EXPLICIT, CONSISTENT, and UNDENIABLE.** Avoid forcing hierarchies onto informational interactions.

        Structure the output as a JSON list of nodes. Each node MUST include:
        - "name": Functionality name (string).
        - "description": Description (string).
        - "parameters": List of parameter names (list of strings or []).
        - "parent_names": List of names of functionalities that MUST be completed immediately before this one based on the rules above. **Use `[]` for root nodes / independent topics / meta-interactions.**

        Rules for Output:
        - Reflect dependencies (or lack thereof) based STRICTLY on consistent conversational evidence and functional necessity.
        - Use the 'name' field as the identifier.
        - Output MUST be valid JSON. Use [] for empty lists.

        Generate the JSON list representing the interaction flow structure:
        """
        # --- End Prompts ---

        # Select the appropriate prompt
        if bot_type == "transactional":
            structuring_prompt = transactional_structuring_prompt
            print("   Using TRANSACTIONAL structuring prompt.")
        else:  # Default to informational
            structuring_prompt = informational_structuring_prompt
            print("   Using INFORMATIONAL structuring prompt.")

        try:
            print("   Asking LLM to determine workflow structure...")
            response = self.llm.invoke(structuring_prompt)
            response_content = response.content

            # Extract JSON from the response
            json_str = None
            json_patterns = [
                r"```json\s*([\s\S]+?)\s*```",  # ```json ... ```
                r"```\s*([\s\S]+?)\s*```",  # ``` ... ```
                r"\[\s*\{.*?\}\s*\]",  # Starts with [ { and ends with } ]
            ]

            for pattern in json_patterns:
                match = re.search(pattern, response_content, re.DOTALL)
                if match:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    break

            if not json_str:  # Fallback if no pattern matched
                if response_content.strip().startswith(
                    "["
                ) and response_content.strip().endswith("]"):
                    json_str = response_content.strip()
                else:
                    raise ValueError("Could not extract JSON block from LLM response.")

            # Clean up potential JSON issues (like trailing commas)
            json_str = re.sub(r"//.*?(\n|$)", "\n", json_str)  # Remove comments
            json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)  # Remove trailing commas
            # Parse the JSON string into a list of node info dicts
            structured_nodes_info = json.loads(json_str)
            if not isinstance(structured_nodes_info, list):
                raise ValueError("LLM response is not a JSON list.")

            # Build the hierarchy from the parent_names info
            nodes_map: Dict[str, Dict[str, Any]] = {  # Map name to node info
                node_info["name"]: node_info
                for node_info in structured_nodes_info
                if "name" in node_info
            }
            # Initialize children list for all nodes
            for node_info in nodes_map.values():
                node_info["children"] = []
            # Link children to parents based on 'parent_names'
            for node_name, node_info in nodes_map.items():
                parent_names = node_info.get("parent_names", [])
                for parent_name in parent_names:
                    if parent_name in nodes_map:
                        parent_node_info = nodes_map[parent_name]
                        # Add child if not already present
                        if node_info not in parent_node_info.get("children", []):
                            parent_node_info.setdefault("children", []).append(
                                node_info
                            )

            # Find root nodes (nodes that are not children of any other node)
            all_child_names = set()
            for node_info in nodes_map.values():
                for child_info in node_info.get("children", []):
                    if isinstance(child_info, dict) and "name" in child_info:
                        all_child_names.add(child_info["name"])

            root_nodes_dicts = [
                node_info
                for node_name, node_info in nodes_map.items()
                if node_name not in all_child_names
            ]
            print(f"   Built structure with {len(root_nodes_dicts)} root node(s).")

            # Update state with the final structured list of dictionaries
            return {**state, "discovered_functionalities": root_nodes_dicts}

        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            print(f"   Error: Failed to decode JSON from LLM response: {e}")
            print(f"   LLM Response Content:\n{response_content}")
            # Keep the original flat list if structuring fails
            return state
        except Exception as e:
            # Handle other errors during structure building
            print(f"   Error during structure building: {e}")
            # Keep the original flat list
            return state
