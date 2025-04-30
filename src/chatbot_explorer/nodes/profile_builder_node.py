from typing import Any

from chatbot_explorer.generation.profile_builder import build_profile_yaml
from chatbot_explorer.schemas.graph_state_model import State


def profile_builder_node(state: State) -> dict[str, Any]:
    """Node that takes all the necessary parameters and builds the YAML.

    Args:
        state (State): The current graph state.

    Returns:
        dict: Updated state dictionary with 'built_profiles'.
    """
    if not state.get("conversation_goals"):
        print("\n--- Skipping profile building: No goals with parameters found. ---")
        return {"built_profiles": []}

    print("\n--- Building user profiles ---")
    built_profiles = []

    # Get fallback message (or use a default)
    fallback_message = state.get("fallback_message", "I'm sorry, I don't understand.")

    # Get primary language (or default to English)
    primary_language = "English"
    if state.get("supported_languages") and len(state["supported_languages"]) > 0:
        primary_language = state["supported_languages"][0]

    # Build YAML for each profile goal set
    for profile in state["conversation_goals"]:
        try:
            # build_profile_yaml expects dict, returns dict/yaml string
            profile_yaml_content = build_profile_yaml(
                profile,
                fallback_message=fallback_message,
                primary_language=primary_language,
            )
            built_profiles.append(profile_yaml_content)
        except Exception as e:
            print(f"Error building profile for goal: {profile.get('name', 'N/A')}. Error: {e}")
            # Optionally skip this profile or add a placeholder error

    # Update state with the list of profile dicts/strings
    return {"built_profiles": built_profiles}
