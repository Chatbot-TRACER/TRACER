"""LangGraph Node for Profile Generation."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.generation.profile_generation import (
    ProfileGenerationConfig,
    generate_profile_content,
)
from chatbot_explorer.schemas.graph_state_model import State


def profile_generator_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node responsible for generating the core content of user profiles.

    This includes:
    - Defining distinct user roles and scenarios based on functionalities.
    - Generating user-centric goals for each scenario.
    - Defining variables ({{variable}}) within goals and their data/types.
    - Generating relevant context points for the simulator.
    - Defining expected output fields to extract from chatbot responses.

    Args:
        state (State): The current graph state, containing discovered functionalities,
                       limitations, conversation history, etc.
        llm: The language model instance used for generation.

    Returns:
        dict[str, Any]: Dictionary containing the generated profile content under the
                        'conversation_goals' key. Each item in the list is a dictionary
                        representing a complete profile (name, role, goals, variables, context, outputs).
    """
    if not state.get("discovered_functionalities"):
        print("\n--- Skipping goal generation: No structured functionalities found. ---")
        return {"conversation_goals": []}

    print("\n--- Generating conversation goals from structured data ---")

    # Functionalities are now dicts (structured from previous node)
    structured_root_dicts: list[dict[str, Any]] = state["discovered_functionalities"]

    # Get workflow structure (which is the structured functionalities itself)
    workflow_structure = structured_root_dicts  # Use the structured data directly

    # Get chatbot type from state
    chatbot_type = state.get("chatbot_type", "unknown")
    print(f"   Chatbot type for goal generation: {chatbot_type}")

    # Helper to get all descriptions from the structure
    def get_all_descriptions(nodes: list[dict[str, Any]]) -> list[str]:
        descriptions = []
        for node in nodes:
            if node.get("description"):
                descriptions.append(node["description"])
            if node.get("children"):
                child_descriptions = get_all_descriptions(node["children"])
                descriptions.extend(child_descriptions)
        return descriptions

    functionality_descriptions = get_all_descriptions(structured_root_dicts)

    if not functionality_descriptions:
        print("   Warning: No descriptions found in structured functionalities.")
        return {"conversation_goals": []}

    print(f" -> Preparing {len(functionality_descriptions)} descriptions (from structure) for goal generation.")

    try:
        # Create the config dictionary
        config: ProfileGenerationConfig = {
            "functionalities": functionality_descriptions,
            "limitations": state.get("discovered_limitations", []),
            "llm": llm,
            "workflow_structure": workflow_structure,
            "conversation_history": state.get("conversation_history", []),
            "supported_languages": state.get("supported_languages", []),
            "chatbot_type": chatbot_type,
        }

        # Call the main generation function with the config dictionary
        profiles_with_goals = generate_profile_content(config)

        print(f" -> Generated {len(profiles_with_goals)} profiles with goals, variables, context, and outputs.")
    except (KeyError, TypeError, ValueError) as e:
        print(f"Error during profile/goal generation orchestration: {e}")
        # Consider more specific error handling or logging
        import traceback

        traceback.print_exc()
        return {"conversation_goals": []}  # Return empty list on error
    else:
        # Update state with the fully generated profiles
        return {"conversation_goals": profiles_with_goals}
