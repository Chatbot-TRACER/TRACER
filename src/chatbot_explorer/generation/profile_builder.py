"""Generates structured output definitions based on user profiles and functionalities and validates them."""

import secrets
from typing import Any

import yaml
from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import AVAILABLE_PERSONALITIES, VARIABLE_PATTERN
from chatbot_explorer.prompts.profile_builder_prompts import get_yaml_fix_prompt
from chatbot_explorer.utils.logging_utils import get_logger
from chatbot_explorer.utils.parsing_utils import extract_yaml
from scripts.validation_script import YamlValidator

logger = get_logger()


def build_profile_yaml(profile: dict[str, Any], fallback_message: str, primary_language: str) -> dict[str, Any]:
    """Create the YAML profile dictionary structure from a profile spec.

    Args:
        profile: Profile data including goals and parameters
        fallback_message: The chatbot's fallback message
        primary_language: Primary language for the user

    Returns:
        Dict containing the structured YAML profile
    """
    # Find all {{variables}} used in the string goals
    used_variables = set()
    original_goals_list = profile.get("goals", [])

    for goal_item in original_goals_list:
        if isinstance(goal_item, str):
            variables_in_string_goal = VARIABLE_PATTERN.findall(goal_item)
            used_variables.update(variables_in_string_goal)

    # Create the goals list for YAML (mix of strings and variable dicts)
    yaml_goals = list(profile.get("goals", []))
    yaml_goals.extend({var_name: profile[var_name]} for var_name in used_variables if var_name in profile)

    # Debug the variables and their values
    if logger.isEnabledFor(10):  # DEBUG level
        logger.debug("Building YAML for profile: %s", profile.get("name", "Unnamed"))
        logger.debug("Used variables: %s", used_variables)
        for var_name in used_variables:
            if var_name in profile:
                logger.debug(" → %s: %s", var_name, profile[var_name])

    # Build the chatbot section
    chatbot_section = {
        "is_starter": False,  # Assuming chatbot doesn't start
        "fallback": fallback_message,
    }
    if "outputs" in profile:  # Add expected outputs if any
        chatbot_section["output"] = profile["outputs"]

    # Build the user context list
    user_context = []

    # Define the probability of including a personality
    personality_probability = 75

    # Include a personality based on the defined probability
    if secrets.randbelow(100) < personality_probability:
        selected_personality = secrets.choice(AVAILABLE_PERSONALITIES)
        user_context.append(f"personality: personalities/{selected_personality}")

    # Choose a random temperature
    temperature = round(secrets.choice(range(30, 101)) / 100, 1)

    # Add other context items
    context = profile.get("context", [])
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
