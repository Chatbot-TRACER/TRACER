"""Generates structured output definitions based on user profiles and functionalities and validates them."""

import secrets
from typing import Any

import yaml
from langchain_core.language_models import BaseLanguageModel

from tracer.constants import AVAILABLE_PERSONALITIES, VARIABLE_PATTERN
from tracer.prompts.profile_builder_prompts import get_yaml_fix_prompt
from tracer.scripts.validation_script import YamlValidator
from tracer.utils.logging_utils import get_logger
from tracer.utils.parsing_utils import extract_yaml

logger = get_logger()


def _extract_variables_and_clean_goals(goals_list: list[Any]) -> tuple[set[str], list[Any], str | None]:
    """Extract variables from goals and clean the goals list.

    Args:
        goals_list: Original goals list from profile

    Returns:
        Tuple of (used_variables, cleaned_goals, existing_name_var_def)
    """
    used_variables = set()
    cleaned_goals = []
    existing_name_var_def = None

    for goal_item in goals_list:
        if isinstance(goal_item, dict) and "name" in goal_item:
            # Save the existing name variable definition
            existing_name_var_def = goal_item["name"]
        else:
            cleaned_goals.append(goal_item)
            # If it's a string goal, collect variables
            if isinstance(goal_item, str):
                variables_in_string_goal = VARIABLE_PATTERN.findall(goal_item)
                used_variables.update(variables_in_string_goal)

    return used_variables, cleaned_goals, existing_name_var_def


def _build_yaml_goals(
    cleaned_goals: list[Any], used_variables: set[str], profile: dict[str, Any], existing_name_var_def: str | None
) -> list[Any]:
    """Build the goals list for YAML with variable definitions.

    Args:
        cleaned_goals: Goals list without name variable definitions
        used_variables: Set of variables found in string goals
        profile: Original profile data
        existing_name_var_def: Existing name variable definition if any

    Returns:
        Complete goals list for YAML
    """
    yaml_goals = cleaned_goals.copy()

    # Clean up the profile by removing any variable definition that might have the profile name
    profile_for_variables = {k: v for k, v in profile.items() if k != "name" or k not in used_variables}

    # Add all variable definitions except "name"
    yaml_goals.extend(
        {var_name: profile_for_variables[var_name]} for var_name in used_variables if var_name in profile_for_variables
    )

    # Add the name variable definition if it exists in the original goals
    if existing_name_var_def:
        yaml_goals.append({"name": existing_name_var_def})

    return yaml_goals


def _build_chatbot_section(profile: dict[str, Any], fallback_message: str) -> dict[str, Any]:
    """Build the chatbot section of the YAML profile.

    Args:
        profile: Profile data
        fallback_message: The chatbot's fallback message

    Returns:
        Chatbot section dictionary
    """
    chatbot_section = {
        "is_starter": False,  # Assuming chatbot doesn't start
        "fallback": fallback_message,
    }
    if "outputs" in profile:  # Add expected outputs if any
        chatbot_section["output"] = profile["outputs"]

    return chatbot_section


def _build_user_context(profile: dict[str, Any]) -> list[str]:
    """Build the user context list with personality and other context items.

    Args:
        profile: Profile data containing context

    Returns:
        List of user context items
    """
    user_context = []

    # Define the probability of including a personality
    personality_probability = 75

    # Include a personality based on the defined probability
    if secrets.randbelow(100) < personality_probability:
        selected_personality = secrets.choice(AVAILABLE_PERSONALITIES)
        user_context.append(f"personality: personalities/{selected_personality}")

    # Add other context items
    context = profile.get("context", [])
    if isinstance(context, str):
        user_context.append(context)
    else:
        user_context.extend(context)

    return user_context


def build_profile_yaml(profile: dict[str, Any], fallback_message: str, primary_language: str) -> dict[str, Any]:
    """Create the YAML profile dictionary structure from a profile spec.

    Args:
        profile: Profile data including goals and parameters
        fallback_message: The chatbot's fallback message
        primary_language: Primary language for the user

    Returns:
        Dict containing the structured YAML profile
    """
    original_goals_list = profile.get("goals", [])
    profile_name = profile.get("name", "Unnamed")

    # Extract variables and clean goals
    used_variables, cleaned_goals, existing_name_var_def = _extract_variables_and_clean_goals(original_goals_list)

    # Build YAML goals with variable definitions
    yaml_goals = _build_yaml_goals(cleaned_goals, used_variables, profile, existing_name_var_def)

    # Debug logging
    if logger.isEnabledFor(10):
        logger.debug("Building YAML for profile: %s", profile_name)
        logger.debug("Used variables: %s", used_variables)
        for var_name in used_variables:
            if var_name in profile:
                logger.debug(" → %s: %s", var_name, profile[var_name])

    # Build sections
    chatbot_section = _build_chatbot_section(profile, fallback_message)
    user_context = _build_user_context(profile)

    # Choose a random temperature
    temperature = round(secrets.choice(range(30, 101)) / 100, 1)

    # Get conversation settings
    conversation_section = profile.get("conversation", {})

    # Assemble the final profile dictionary
    return {
        "test_name": profile_name,
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


def validate_and_fix_profile(
    profile: dict[str, Any], validator: YamlValidator, llm: BaseLanguageModel
) -> dict[str, Any]:
    """Validate a profile and try to fix it using LLM if needed."""
    # Convert profile dict to YAML string for validation
    yaml_content = yaml.dump(profile, sort_keys=False, allow_unicode=True)
    errors = validator.validate(yaml_content)  # Validate

    profile_name = profile.get("test_name", "Unnamed profile")

    if not errors:
        # Profile is valid
        logger.info(" ✅ Profile '%s' valid, no fixes needed.", profile_name)
        return profile

    # Profile has errors
    error_count = len(errors)
    logger.warning(" ⚠️ Profile '%s' has %d validation errors", profile_name, error_count)

    max_errors_to_print = 3
    # Log first few errors
    for e in errors[:max_errors_to_print]:
        logger.warning("  • %s: %s", e.path, e.message)
    if error_count > max_errors_to_print:
        logger.warning("  • ... and %d more errors", error_count - max_errors_to_print)

    # Prepare prompt for LLM to fix errors
    error_messages = "\n".join(f"- {e.path}: {e.message}" for e in errors)

    fix_prompt = get_yaml_fix_prompt(error_messages, yaml_content)

    logger.verbose("  Asking LLM to fix profile '%s'...", profile_name)

    try:
        # Ask LLM to fix it
        fixed_yaml_response = llm.invoke(fix_prompt)
        fixed_yaml_str = fixed_yaml_response.content

        # Extract and parse the fixed YAML
        just_yaml = extract_yaml(fixed_yaml_str)
        if not just_yaml:
            logger.warning("  ✗ LLM response did not contain a YAML block.")
            return profile  # Keep original

        fixed_profile = yaml.safe_load(just_yaml)

        # Re-validate the fixed YAML
        re_errors = validator.validate(just_yaml)

        if not re_errors:
            # Fixed successfully!
            logger.info("  ✓ Profile '%s' fixed successfully!", profile_name)
            return fixed_profile

        # Still has errors, keep original
        logger.warning("  ✗ LLM couldn't fix all errors (%d remaining)", len(re_errors))
        for e in re_errors[:max_errors_to_print]:
            logger.debug("    • %s: %s", e.path, e.message)

    except yaml.YAMLError:
        logger.exception("  ✗ Failed to parse fixed YAML for '%s'", profile_name)
        return profile
    except Exception:
        logger.exception("  ✗ Unexpected error fixing profile '%s'", profile_name)

    return profile  # Keep original
