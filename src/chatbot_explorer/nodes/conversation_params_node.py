"""Node for generating conversation parameters (number, cost, style) for user profiles."""

import contextlib
import random
from typing import Any, TypedDict

from langchain_core.language_models.base import BaseLanguageModel

from chatbot_explorer.prompts.conversation_params_prompts import (
    PromptLanguageSupport,
    PromptPreviousParams,
    PromptProfileContext,
    get_goal_style_prompt,
    get_interaction_style_prompt,
    get_max_cost_prompt,
    get_number_prompt,
)
from chatbot_explorer.schemas.graph_state_model import State
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_DISPLAYED_STYLES = 2

# Global constants for deterministic parameter generation
DEFAULT_RUNS_NO_VARIABLES = 3
DEFAULT_RUNS_NON_FORWARD_VARIABLES = 3
BASE_COST_PER_CONVERSATION = 0.15
MIN_GOAL_LIMIT = 15
MAX_GOAL_LIMIT = 30

# Available interaction styles for randomization
AVAILABLE_INTERACTION_STYLES = [
    "long phrases",
    "change your mind",
    "make spelling mistakes",
    "single question",
    "all questions"
]

# --- Helper Functions for extract_profile_variables ---


def _get_profile_variables(profile: dict[str, Any]) -> list[str]:
    """Extracts all defined variable names from a profile."""
    variables = []

    # Check for variables at the top level (original behavior)
    variables.extend([
        var_name
        for var_name, var_def in profile.items()
        if isinstance(var_def, dict) and "function" in var_def and "data" in var_def
    ])

    # Also check for variables nested within the 'goals' list
    if "goals" in profile and isinstance(profile["goals"], list):
        for item in profile["goals"]:
            # If the goal item itself is a dictionary with 'function' and 'data'
            if isinstance(item, dict) and "function" in item and "data" in item:
                # Try to find a name for this variable
                for key, value in item.items():
                    if key not in ["function", "data", "type"]:
                        variables.append(key)
                        break

            # If the goal item is a dictionary with key-value pairs where values are variable definitions
            elif isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, dict) and "function" in value and "data" in value:
                        variables.append(key)

    return variables


def _get_variable_def(profile: dict[str, Any], var_name: str) -> dict | None:
    """Gets the variable definition from the profile, checking both top level and within goals."""
    # Check at top level first
    var_def = profile.get(var_name)
    if isinstance(var_def, dict) and "function" in var_def and "data" in var_def:
        return var_def

    # Check within goals if not found at top level
    if "goals" in profile and isinstance(profile["goals"], list):
        for item in profile["goals"]:
            # If the goal item is a dictionary with key-value pairs
            if isinstance(item, dict):
                # Check if the item itself is a variable definition with the matching name
                for key, value in item.items():
                    if key == var_name and isinstance(value, dict) and "function" in value and "data" in value:
                        return value

                # Or if the item directly has the var_name key
                if var_name in item and isinstance(item[var_name], dict) and "function" in item[var_name] and "data" in item[var_name]:
                    return item[var_name]

    return None


def _calculate_combinations(profile: dict[str, Any], variables: list[str]) -> int:
    """Calculates the potential number of combinations based on variable definitions.

    This function determines how many different combinations could be created from
    the variables in a profile, taking into account list sizes and range-defined values.
    """
    combinations = 1
    var_sizes = {}

    # Calculate the size of each variable
    for var_name in variables:
        var_def = _get_variable_def(profile, var_name)
        if var_def and "data" in var_def:
            data = var_def.get("data", [])
            var_size = 1  # Default size

            if isinstance(data, list):
                var_size = len(data) if data else 1
            elif isinstance(data, dict) and all(k in data for k in ["min", "max", "step"]) and data["step"] != 0:
                steps = (data["max"] - data["min"]) / data["step"] + 1
                var_size = int(steps) if steps > 0 else 1

            var_sizes[var_name] = var_size
            combinations *= var_size

    # If we have forward dependencies, adjust the combinations calculation
    if "forward_dependencies" in profile:
        forward_dependencies = profile["forward_dependencies"]

        # For each dependent variable, determine if its size is already accounted for
        for dependent_var, source_vars in forward_dependencies.items():
            # Skip variables that couldn't be sized
            if dependent_var not in var_sizes:
                continue

            # If the dependent variable depends on variables we've already counted,
            # we need to avoid double-counting those combinations
            if any(source_var in var_sizes for source_var in source_vars):
                # This approach is simplified - in complex cases with nested dependencies,
                # we would need a more sophisticated graph analysis
                combinations = combinations // var_sizes[dependent_var]

    # Additional check for nested forward references in custom format
    for var_name in variables:
        var_def = _get_variable_def(profile, var_name)
        if var_def and "function" in var_def:
            func = var_def["function"]
            if "forward" in func and "(" in func and ")" in func:
                param = func.split("(")[1].split(")")[0]
                if param and param != "rand" and not param.isdigit() and param in var_sizes:
                    # This variable depends on another, adjust to avoid double-counting
                    if var_name in var_sizes:
                        combinations = combinations // var_sizes[var_name]

    return max(combinations, 1)  # Ensure at least 1 combination


def _check_nested_forwards(profile: dict[str, Any], variables: list[str]) -> tuple[bool, list[str], str]:
    """Checks for nested forward dependencies and calculates related info."""
    has_nested_forwards = profile.get("has_nested_forwards", False)
    forward_with_dependencies = []
    nested_forward_info = ""

    if "forward_dependencies" in profile:
        forward_dependencies = profile["forward_dependencies"]
        forward_with_dependencies = list(forward_dependencies.keys())

        if has_nested_forwards and "nested_forward_chains" in profile:
            nested_chains = profile["nested_forward_chains"]
            chain_descriptions = [f"Chain: {' → '.join(chain)}" for chain in nested_chains]

            if chain_descriptions:
                nested_forward_info = "\nNested dependency chains detected:\n" + "\n".join(chain_descriptions)
                combinations = _calculate_combinations(profile, variables)
                nested_forward_info += f"\nPotential combinations: approximately {combinations}"
    else:  # Fallback if structured dependencies aren't present
        # Check for forward dependencies in variable definitions
        for var_name in variables:
            var_def = _get_variable_def(profile, var_name)
            if var_def and "function" in var_def:
                func = var_def["function"]
                if "forward" in func and "(" in func and ")" in func:
                    param = func.split("(")[1].split(")")[0]
                    if param and param != "rand" and not param.isdigit():
                        forward_with_dependencies.append(var_name)
                        # If the referenced parameter is itself a forward, that's a nested forward
                        ref_var_def = _get_variable_def(profile, param)
                        if ref_var_def and "function" in ref_var_def and "forward" in ref_var_def["function"]:
                            has_nested_forwards = True

    return has_nested_forwards, forward_with_dependencies, nested_forward_info


def _build_variables_info_string(
    variables: list[str],
    forward_with_dependencies: list[str],
    nested_forward_info: str,
    *,
    has_nested_forwards: bool,
) -> str:
    """Builds the descriptive string about variables for LLM prompts."""
    if not variables:
        return ""

    variables_info = f"\nThis profile has {len(variables)} variables: {', '.join(variables)}"
    if forward_with_dependencies:
        variables_info += (
            f"\n{len(forward_with_dependencies)} variables have dependencies: {', '.join(forward_with_dependencies)}"
        )
        if has_nested_forwards:
            variables_info += "\nThis creates COMBINATIONS that could be explored with 'all_combinations', 'sample(X)', or a fixed number."
            variables_info += f"\nIMPORTANT: This profile has NESTED FORWARD DEPENDENCIES.{nested_forward_info}"
    return variables_info


def extract_profile_variables(profile: dict[str, Any]) -> tuple[list[str], list[str], bool, str, str]:
    """Extracts variables, dependency info, and builds a descriptive string from a profile.

    Args:
        profile: The user profile dictionary.

    Returns:
        A tuple containing:
            - List of all variable names.
            - List of variables with forward dependencies.
            - Boolean indicating if nested forwards exist.
            - String with details about nested forward chains and combinations.
            - A combined descriptive string about variables for LLM prompts.
    """
    variables = _get_profile_variables(profile)
    has_nested_forwards, forward_with_dependencies, nested_forward_info = _check_nested_forwards(profile, variables)
    variables_info = _build_variables_info_string(
        variables, forward_with_dependencies, nested_forward_info, has_nested_forwards=has_nested_forwards
    )
    return variables, forward_with_dependencies, has_nested_forwards, nested_forward_info, variables_info


# --- Language Info Preparation ---


def prepare_language_info(supported_languages: list[str] | None) -> tuple[str, str, str]:
    """Prepares language-related strings for LLM prompts."""
    language_info = ""
    languages_example = ""
    supported_languages_text = ""

    if supported_languages:
        language_info = f"\nSUPPORTED LANGUAGES: {', '.join(supported_languages)}"
        supported_languages_text = f"({', '.join(supported_languages)})"
        languages_example = "\n".join([f"- {lang.lower()}" for lang in supported_languages])

    return language_info, languages_example, supported_languages_text


# --- Context for Parameter Requests ---


class ParamRequestContext(TypedDict):
    """Context dictionary for requesting conversation parameters."""

    llm: BaseLanguageModel
    profile: dict[str, Any]
    variables_info: str
    language_info: str
    supported_languages_text: str
    languages_example: str


# --- LLM Request Functions ---


def _parse_number_response(response_text: str, default_number: str | int) -> str | int:
    """Parses the LLM response for the 'number' parameter."""
    extracted_number = None
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "number":
            value = value.strip()
            if value == "all_combinations":
                extracted_number = "all_combinations"
            elif "sample" in value.lower() and "(" in value and ")" in value:
                with contextlib.suppress(ValueError):
                    sample_value = float(value.split("(")[1].split(")")[0])
                    min_sample_value = 0.1
                    max_sample_value = 1.0
                    if min_sample_value <= sample_value <= max_sample_value:
                        extracted_number = f"sample({sample_value})"
            elif value.isdigit():
                extracted_number = int(value)
            break  # Found number, no need to check further lines
    return extracted_number if extracted_number is not None else default_number


def _get_max_variable_size(profile: dict[str, Any]) -> int:
    """Determines the maximum size of any variable list in the profile."""
    max_size = 1  # Default minimum size
    variables = _get_profile_variables(profile)

    for var_name in variables:
        var_def = _get_variable_def(profile, var_name)
        if var_def and "data" in var_def:
            data = var_def.get("data", [])
            if isinstance(data, list):
                current_size = len(data)
                if current_size > max_size:
                    max_size = current_size
            elif isinstance(data, dict) and all(k in data for k in ["min", "max", "step"]) and data["step"] != 0:
                # Handle range-defined variables
                steps = (data["max"] - data["min"]) / data["step"] + 1
                current_size = int(steps) if steps >= 1 else 1
                if current_size > max_size:
                    max_size = current_size

    return max_size


def request_number_from_llm(context: ParamRequestContext, variables: list[str]) -> str | int:
    """Prompts LLM to determine the NUMBER parameter."""
    # Get the maximum variable size for better default calculation
    max_var_size = _get_max_variable_size(context["profile"])

    # Set a smarter default based on variables and their sizes
    if context["has_nested_forwards"]:
        total_combinations = _calculate_combinations(context["profile"], variables)
        if total_combinations < 5:
            default_number = "all_combinations"
        else:
            default_number = f"sample(0.2)"
    else:
        default_number = max(max_var_size, 2) if variables else 2

    prompt = get_number_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        has_nested_forwards=context["has_nested_forwards"],
    )
    response = context["llm"].invoke(prompt)

    # Parse response with the more intelligent default
    llm_suggested_number = _parse_number_response(response.content.strip(), default_number)

    # For profiles with variables, ensure the number is at least the maximum variable size
    # unless it's a special value like "all_combinations" or "sample(X)"
    if variables and isinstance(llm_suggested_number, int) and max_var_size > 3:
        # If the LLM chose a number smaller than the max variable size, use the max size instead
        # This ensures adequate coverage of all variable options
        if llm_suggested_number < max_var_size:
            return max_var_size

    return llm_suggested_number


def request_max_cost_from_llm(context: ParamRequestContext, number_value: str | int) -> float:
    """Prompts LLM to determine the MAX_COST parameter."""
    prompt = get_max_cost_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        number_value=number_value,
    )
    response = context["llm"].invoke(prompt)
    response_text = response.content.strip()
    default_cost = 1.0 if not context["has_nested_forwards"] else 2.0
    max_cost = default_cost
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "max_cost":
            with contextlib.suppress(ValueError):
                max_cost = float(value.strip())
            break
    return max_cost


def _parse_goal_style_response(response_text: str) -> dict[str, Any]:
    """Parses the LLM response for the 'goal_style' parameter."""
    goal_style = {"steps": 11}  # Default
    selected_style = "steps"
    goal_style_value = 11

    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "goal_style" and value in ["steps", "all_answered", "random_steps"]:
            selected_style = value
        elif key == "goal_style_value":
            with contextlib.suppress(ValueError):
                goal_style_value = int(value)

        # Update goal_style based on potentially updated selected_style and goal_style_value
        if selected_style == "steps":
            goal_style = {"steps": goal_style_value}
        elif selected_style == "all_answered":
            goal_style = {"all_answered": {"export": False, "limit": goal_style_value}}
        elif selected_style == "random_steps":
            goal_style = {"random_steps": goal_style_value}

    return goal_style


def request_goal_style_from_llm(
    context: ParamRequestContext, number_value: str | int, max_cost: float
) -> dict[str, Any]:
    """Prompts LLM to determine the GOAL_STYLE parameter."""
    prompt = get_goal_style_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        number_value=number_value,
        max_cost=max_cost,
    )
    response = context["llm"].invoke(prompt)
    return _parse_goal_style_response(response.content.strip())


def request_interaction_style_from_llm(
    context: ParamRequestContext, number_value: str | int, max_cost: float, goal_style: dict[str, Any]
) -> list[str]:
    """Prompts LLM to determine the INTERACTION_STYLE parameter."""
    profile_context: PromptProfileContext = {
        "profile": context["profile"],
        "variables_info": context["variables_info"],
        "language_info": context["language_info"],
    }
    prev_params: PromptPreviousParams = {
        "number_value": number_value,
        "max_cost": max_cost,
        "goal_style": goal_style,
    }
    lang_support: PromptLanguageSupport = {
        "supported_languages_text": context["supported_languages_text"],
        "languages_example": context["languages_example"],
    }

    # Call the prompt function with the structured arguments
    prompt = get_interaction_style_prompt(
        profile_context=profile_context,
        prev_params=prev_params,
        lang_support=lang_support,
    )
    response = context["llm"].invoke(prompt)
    response_text = response.content.strip()
    interaction_styles = []
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "interaction_style":
            value = value.strip()
            if "[" in value and "]" in value:
                styles_part = value.replace("[", "").replace("]", "")
                styles = [s.strip().strip("\"'") for s in styles_part.split(",")]
                interaction_styles = [s for s in styles if s]
            else:
                style = value.strip().strip("\"'")
                if style:
                    interaction_styles.append(style)
            break
    return interaction_styles


# --- Main Generation Function ---


def generate_conversation_parameters(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generates conversation parameters for each profile using sequential LLM prompting.

    Args:
        profiles: List of user profile dictionaries.
        llm: The language model instance.
        supported_languages: Optional list of supported languages.

    Returns:
        The list of profiles with an added 'conversation' key containing the generated parameters.
    """
    language_info, languages_example, supported_languages_text = prepare_language_info(supported_languages)

    # Process each profile
    total_profiles = len(profiles)
    for i, profile in enumerate(profiles, 1):
        profile_name = profile.get("name", f"Profile {i}")

        # Extract variables information
        variables, forward_vars, has_nested_forwards, _, variables_info = extract_profile_variables(profile)

        # Calculate potential combinations for more informed decisions
        max_var_size = _get_max_variable_size(profile)
        total_combinations = _calculate_combinations(profile, variables)

        if variables:
            var_summary = ", ".join(variables)
            logger.debug("Profile has %d variables: %s", len(variables), var_summary)
            logger.debug(
                "Maximum variable size: %d, potential combinations: %d",
                max_var_size,
                total_combinations,
            )

            if forward_vars:
                logger.debug(
                    "With %d dependent variables: %s",
                    len(forward_vars),
                    ", ".join(forward_vars),
                )
        else:
            logger.debug("Profile has no variables")

        # Create context for this profile's requests
        request_context: ParamRequestContext = {
            "llm": llm,
            "profile": profile,
            "variables_info": variables_info,
            "language_info": language_info,
            "has_nested_forwards": has_nested_forwards,
            "supported_languages_text": supported_languages_text,
            "languages_example": languages_example,
        }

        # Sequential prompting with minimal but sufficient logging
        logger.debug("Determining parameters for profile '%s'", profile_name)

        # Fetch parameters sequentially with minimal logging
        number_value = request_number_from_llm(request_context, variables)
        max_cost = request_max_cost_from_llm(request_context, number_value)
        goal_style = request_goal_style_from_llm(request_context, number_value, max_cost)
        interaction_styles = request_interaction_style_from_llm(request_context, number_value, max_cost, goal_style)

        # Log a concise summary of the key parameters
        goal_style_type = next(iter(goal_style.keys())) if goal_style else "unknown"
        interaction_style_summary = ""
        if interaction_styles:
            interaction_style_summary = f", styles: {', '.join(interaction_styles[:MAX_DISPLAYED_STYLES])}" + (
                f" +{len(interaction_styles) - MAX_DISPLAYED_STYLES} more"
                if len(interaction_styles) > MAX_DISPLAYED_STYLES
                else ""
            )

        logger.info(" ✅ Generated conversation parameters %d/%d: '%s'", i, total_profiles, profile_name)

        # Log detailed information about number selection based on variables
        if variables:
            number_reason = ""
            if number_value == "all_combinations":
                number_reason = f" (testing all {total_combinations} combinations)"
            elif isinstance(number_value, str) and "sample" in number_value:
                sample_pct = number_value.split("(")[1].split(")")[0]
                approx_count = round(float(sample_pct) * total_combinations)
                number_reason = f" (sampling ~{approx_count} of {total_combinations} combinations)"
            elif isinstance(number_value, int) and number_value >= max_var_size:
                number_reason = f" (adequate to cover largest variable with {max_var_size} options)"

            logger.debug(
                "Parameters: number=%s%s, cost=%.2f, goal=%s%s",
                number_value,
                number_reason,
                max_cost,
                goal_style_type,
                interaction_style_summary,
            )
        else:
            logger.debug(
                "Parameters: number=%s, cost=%.2f, goal=%s%s",
                number_value,
                max_cost,
                goal_style_type,
                interaction_style_summary,
            )

        # Build and assign parameters
        conversation_params = {"number": number_value, "max_cost": max_cost, "goal_style": goal_style}
        if interaction_styles:
            conversation_params["interaction_style"] = interaction_styles
        profile["conversation"] = conversation_params

    return profiles


# --- Deterministic Conversation Parameters Generation ---


def generate_deterministic_parameters(
    profiles: list[dict[str, Any]], supported_languages: list[str] | None = None
) -> list[dict[str, Any]]:
    """Generates conversation parameters deterministically without using LLM calls.

    Args:
        profiles: List of user profile dictionaries.
        supported_languages: Optional list of supported languages.

    Returns:
        The list of profiles with an added 'conversation' key containing the generated parameters.
    """
    # Process each profile
    total_profiles = len(profiles)

    for i, profile in enumerate(profiles, 1):
        profile_name = profile.get("name", f"Profile {i}")

        # Extract variables information
        variables, forward_vars, has_nested_forwards, _, _ = extract_profile_variables(profile)

        # Get the maximum variable size
        max_var_size = _get_max_variable_size(profile)

        # Count goals for limit calculation
        num_goals = 0
        num_outputs = 0
        if "goals" in profile and isinstance(profile["goals"], list):
            num_goals = len([g for g in profile["goals"] if isinstance(g, str)])
        if "output" in profile.get("chatbot", {}) and isinstance(profile["chatbot"]["output"], list):
            num_outputs = len(profile["chatbot"]["output"])

        # Determine number of conversations
        if has_nested_forwards:
            total_combinations = _calculate_combinations(profile, variables)
            if total_combinations < 10:
                number_value = "all_combinations"
            else:
                number_value = f"sample(0.3)"
        elif max_var_size > 1:
            # If we have variables with data, use the maximum size
            number_value = max_var_size
        elif forward_vars:
            # If we have forward variables but no significant max size
            number_value = DEFAULT_RUNS_NON_FORWARD_VARIABLES
        else:
            # Default for simple profiles
            number_value = DEFAULT_RUNS_NO_VARIABLES

        # Calculate max cost based on number of conversations
        if isinstance(number_value, int):
            max_cost = round(BASE_COST_PER_CONVERSATION * number_value, 2)
        elif number_value == "all_combinations":
            # For all combinations, estimate cost based on variable combinations
            total_combinations = _calculate_combinations(profile, variables)
            max_cost = round(BASE_COST_PER_CONVERSATION * min(total_combinations, 10), 2)
        elif isinstance(number_value, str) and "sample" in number_value:
            # For sample, estimate based on the sample percentage and combinations
            sample_ratio = float(number_value.split("(")[1].split(")")[0])
            total_combinations = _calculate_combinations(profile, variables)
            estimated_runs = round(total_combinations * sample_ratio)
            max_cost = round(BASE_COST_PER_CONVERSATION * min(estimated_runs, 10), 2)
        else:
            # Fallback
            max_cost = 1.0

        # Ensure the cost is at least a minimum amount
        max_cost = max(max_cost, 0.5)

        # Always use all_answered goal style with limit based on goals count
        logger.debug(f"Num goals {num_goals}")
        logger.debug(f"Num outputs {num_outputs}")

        goal_limit = min(max(MIN_GOAL_LIMIT, (num_goals + num_outputs) * 2), MAX_GOAL_LIMIT)
        goal_style = {"all_answered": {"export": False, "limit": goal_limit}}

        # Select 1-2 random interaction styles
        num_styles = random.randint(1, 2)
        interaction_styles = random.sample(AVAILABLE_INTERACTION_STYLES, num_styles)


        # Build the conversation parameters
        conversation_params = {
            "number": number_value,
            "max_cost": max_cost,
            "goal_style": goal_style,
            "interaction_style": interaction_styles
        }

        # Log what we've generated
        style_summary = f", styles: {', '.join(interaction_styles[:MAX_DISPLAYED_STYLES])}"
        if len(interaction_styles) > MAX_DISPLAYED_STYLES:
            style_summary += f" +{len(interaction_styles) - MAX_DISPLAYED_STYLES} more"

        logger.info(" ✅ Generated conversation parameters %d/%d: '%s'", i, total_profiles, profile_name)

        if variables:
            logger.debug(
                "Parameters: number=%s (from %d variables), cost=%.2f, goal=all_answered (limit: %d)%s",
                number_value,
                len(variables),
                max_cost,
                goal_limit,
                style_summary,
            )
        else:
            logger.debug(
                "Parameters: number=%s (no variables), cost=%.2f, goal=all_answered (limit: %d)%s",
                number_value,
                max_cost,
                goal_limit,
                style_summary,
            )

        # Assign parameters to the profile
        profile["conversation"] = conversation_params

    return profiles


# --- LangGraph Node ---


def conversation_params_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that generates specific parameters needed for conversation goals."""
    conversation_goals = state.get("conversation_goals")
    if not conversation_goals:
        logger.info("Skipping conversation parameters: No goals generated.")
        return {"conversation_goals": []}

    logger.info("\nStep 3: Conversation parameters generation")
    logger.info("--------------------------\n")

    # Flatten functionalities (currently unused but kept for context)
    structured_root_dicts = state.get("discovered_functionalities", [])
    flat_func_info = []
    nodes_to_process = list(structured_root_dicts)
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        info = {k: v for k, v in node.items() if k != "children"}
        flat_func_info.append(info)
        if node.get("children"):
            nodes_to_process.extend(node["children"])

    try:
        # Initial progress message
        total_profiles = len(conversation_goals)
        logger.info("Generating conversation parameters for %d profiles:\n", total_profiles)

        # Generate parameters deterministically instead of using LLM
        profiles_with_params = generate_deterministic_parameters(
            conversation_goals,
            supported_languages=state.get("supported_languages"),
        )

    except Exception:
        logger.exception("Error during parameter generation")
        return {"conversation_goals": conversation_goals}
    else:
        return {"conversation_goals": profiles_with_params}
