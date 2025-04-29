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
from .session import run_exploration_session

from .utils.conversation.language_detection import extract_supported_languages
from .utils.conversation.fallback_detection import extract_fallback_message
from .utils.conversation.conversation_utils import format_conversation
from .utils.analysis.chatbot_classification import classify_chatbot_type
from .utils.analysis.workflow_builder import build_workflow_structure
from .utils.analysis.profile_generator import (
    build_profile_yaml,
    validate_and_fix_profile,
    extract_yaml,
)

from .state import State


class ChatbotExplorer:
    """Uses LangGraph to explore chatbots."""

    def __init__(self, model_name: str):
        """
        Sets up the explorer.

        Args:
            model_name (str): The name of the OpenAI model to use.
        """
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()

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
                _,
                _,
                updated_roots,
                updated_pending,
                updated_explored,
            ) = run_exploration_session(
                current_session_index,
                max_sessions,
                max_turns,
                self,
                chatbot_connector,
                fallback_message=fallback_message,
                current_node=explore_node,  # None for general exploration
                root_nodes=root_nodes,
                pending_nodes=pending_nodes,
                explored_nodes=explored_nodes,
                supported_languages=supported_languages,
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

        # Classify the bot type first using the imported function with proper arguments
        bot_type = classify_chatbot_type(
            flat_functionality_dicts, conversation_history, self.llm
        )

        # Store the bot type in the state
        state = {**state, "chatbot_type": bot_type}

        # Use the imported build_workflow_structure function
        try:
            structured_nodes = build_workflow_structure(
                flat_functionality_dicts, conversation_history, bot_type, self.llm
            )

            print(f"   Built structure with {len(structured_nodes)} root node(s).")

            # Update state with the final structured list of dictionaries
            return {**state, "discovered_functionalities": structured_nodes}

        except Exception as e:
            # Handle errors during structure building
            print(f"   Error during structure building: {e}")
            # Keep the original flat list
            return state

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
                profile_yaml = build_profile_yaml(
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
                validated_profile = validate_and_fix_profile(
                    profile, validator, self.llm
                )
                validated_profiles.append(validated_profile)

            # Update state with the list of validated profiles
            return {
                "messages": state["messages"],
                "built_profiles": validated_profiles,
            }
        # Return unchanged state if skipped
        return {"messages": state["messages"]}
