from typing import Any, Dict

from scripts.validation_script import YamlValidator

from ..analysis.profile_generator import validate_and_fix_profile
from ..schemas.state import State


def profile_validator_node(state: State, llm) -> Dict[str, Any]:
    """Node that validates generated YAML profiles and tries to fix them using LLM if needed.

    Args:
        state (State): The current graph state.
        llm: The language model instance.

    Returns:
        dict: Updated state dictionary with validated (and potentially fixed) 'built_profiles'.
    """
    if not state.get("built_profiles"):
        print("\n--- Skipping profile validation: No profiles built. ---")
        return {"built_profiles": []}

    print("\n--- Validating user profiles ---")
    validator = YamlValidator()  # Our validator class
    validated_profiles = []  # List to hold good profiles

    for profile_content in state["built_profiles"]:
        try:
            # validate_and_fix_profile takes the content (dict/string), validator, llm
            validated_profile = validate_and_fix_profile(profile_content, validator, llm)
            if validated_profile:  # Only add if validation/fixing was successful
                validated_profiles.append(validated_profile)
            else:
                print("  - Profile failed validation and could not be fixed.")
        except Exception as e:
            print(f"Error during profile validation/fixing: {e}")
            # Optionally skip this profile

    # Update state with the list of validated profiles
    return {"built_profiles": validated_profiles}
