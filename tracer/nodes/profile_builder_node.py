"""LangGraph Node that builds the YAML profiles for the chatbot."""

from typing import Any

from tracer.generation.profile_builder import build_profile_yaml
from tracer.schemas.graph_state_model import State
from tracer.utils.logging_utils import get_logger

logger = get_logger()


def profile_builder_node(state: State) -> dict[str, Any]:
    """Node that takes all the necessary parameters and builds the YAML.

    Args:
        state (State): The current graph state.

    Returns:
        dict: Updated state dictionary with 'built_profiles'.
    """
    if not state.get("conversation_goals"):
        logger.warning("Skipping profile building: No goals with parameters found")
        return {"built_profiles": []}

    logger.info("\nStep 4: Building user profiles")
    logger.info("--------------------------\n")

    built_profiles = []

    # Get fallback message (or use a default)
    fallback_message = state.get("fallback_message", "I'm sorry, I don't understand.")

    # Get primary language (or default to English)
    primary_language = "English"
    if state.get("supported_languages") and len(state["supported_languages"]) > 0:
        primary_language = state["supported_languages"][0]

    # Get model from state (or use default if not provided)
    model = state.get("model", "gpt-4o-mini")

    total_profiles = len(state["conversation_goals"])

    # Build YAML for each profile goal set
    successful = 0
    for i, profile in enumerate(state["conversation_goals"], 1):
        profile_name = profile.get("name", f"Profile {i}")
        logger.debug("Building profile %d/%d: '%s'", i, total_profiles, profile_name)

        try:
            # build_profile_yaml expects dict, returns dict/yaml string
            profile_yaml_content = build_profile_yaml(
                profile,
                fallback_message=fallback_message,
                primary_language=primary_language,
                model=model,
            )
            built_profiles.append(profile_yaml_content)
            successful += 1

        except (KeyError, ValueError, TypeError):
            logger.exception("Error building profile for '%s'", profile_name)

    # Update state with the list of profile dicts/strings
    return {"built_profiles": built_profiles}
