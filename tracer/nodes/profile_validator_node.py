"""LangGraph node to validate and fix YAML profiles."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.generation.profile_builder import validate_and_fix_profile
from tracer.schemas.graph_state_model import State
from tracer.scripts.validation_script import YamlValidator
from tracer.utils.logging_utils import get_logger

logger = get_logger()


def profile_validator_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that validates generated YAML profiles and tries to fix them using LLM if needed.

    Args:
        state (State): The current graph state.
        llm: The language model instance.

    Returns:
        dict: Updated state dictionary with validated (and potentially fixed) 'built_profiles'.
    """
    if not state.get("built_profiles"):
        logger.warning("Skipping profile validation: No profiles built")
        return {"built_profiles": []}

    # Don't log validation header here - it's already logged in profile_builder_node
    # This avoids the duplication in the logs

    validator = YamlValidator()  # Our validator class
    validated_profiles = []  # List to hold good profiles

    profile_count = len(state["built_profiles"])
    logger.debug("Starting validation of %d profiles", profile_count)

    logger.info("Validating %d profiles:", profile_count)

    for i, profile_content in enumerate(state["built_profiles"], 1):
        profile_name = profile_content.get("name", f"Profile {i}")

        try:
            # validate_and_fix_profile takes the content (dict/string), validator, llm
            validated_profile = validate_and_fix_profile(profile_content, validator, llm)

            if validated_profile:  # Only add if validation/fixing was successful
                validated_profiles.append(validated_profile)
                # Don't log successful validation here - already done in profile_builder
                logger.debug("Profile '%s' passed validation", profile_name)
            else:
                logger.warning("Profile '%s' failed validation and could not be fixed", profile_name)

        except (ValueError, TypeError, KeyError):
            logger.exception("Error during validation of profile '%s'", profile_name)

    successful_count = len(validated_profiles)

    # Only log a summary if there were any validation failures
    if successful_count < profile_count:
        logger.info("Validation complete: %d/%d profiles passed", successful_count, profile_count)

    # Update state with the list of validated profiles
    return {"built_profiles": validated_profiles}
