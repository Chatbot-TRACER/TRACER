import random
from typing import Any

import yaml

from chatbot_explorer.constants import AVAILABLE_PERSONALITIES, VARIABLE_PATTERN
from chatbot_explorer.prompts.profile_builder_prompts import get_yaml_fix_prompt
from chatbot_explorer.utils.parsing_utils import extract_yaml


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

    # 75% chance to include a personality
    if random.random() < 0.75:
        selected_personality = random.choice(AVAILABLE_PERSONALITIES)
        user_context.append(f"personality: personalities/{selected_personality}")

    # Choose a random temperature
    temperature = round(random.uniform(0.3, 1.0), 1)

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


def validate_and_fix_profile(profile: dict[str, Any], validator, llm) -> dict[str, Any]:
    """Validate a profile and try to fix it using LLM if needed."""
    # Convert profile dict to YAML string for validation
    yaml_content = yaml.dump(profile, sort_keys=False, allow_unicode=True)
    errors = validator.validate(yaml_content)  # Validate

    if not errors:
        # Profile is valid
        print(f"✓ Profile '{profile['test_name']}' valid, no fixes needed.")
        return profile
    # Profile has errors
    error_count = len(errors)
    print(f"\n⚠ Profile '{profile['test_name']}' has {error_count} validation errors")

    # Print first few errors
    for e in errors[:3]:
        print(f"  • {e.path}: {e.message}")
    if error_count > 3:
        print(f"  • ... and {error_count - 3} more errors")

    # Prepare prompt for LLM to fix errors
    error_messages = "\n".join(f"- {e.path}: {e.message}" for e in errors)

    fix_prompt = get_yaml_fix_prompt(error_messages, yaml_content)

    print("  Asking LLM to fix the profile...")

    try:
        # Ask LLM to fix it
        fixed_yaml_response = llm.invoke(fix_prompt)
        fixed_yaml_str = fixed_yaml_response.content

        # Extract and parse the fixed YAML
        just_yaml = extract_yaml(fixed_yaml_str)
        if not just_yaml:
            print("  ✗ LLM response did not contain a YAML block.")
            return profile  # Keep original

        fixed_profile = yaml.safe_load(just_yaml)

        # Re-validate the fixed YAML
        re_errors = validator.validate(just_yaml)

        if not re_errors:
            # Fixed successfully!
            print("  ✓ Profile fixed successfully!")
            return fixed_profile
        # Still has errors, keep original
        print(f"  ✗ LLM couldn't fix all errors ({len(re_errors)} remaining)")
        return profile  # Keep original
    except yaml.YAMLError as e:
        print(f"  ✗ Failed to parse fixed YAML: {e}")
        return profile  # Keep original
    except Exception as e:
        # LLM call failed or other error, keep original
        print(f"  ✗ Failed to fix profile: {type(e).__name__} - {e}")
        return profile  # Keep original
