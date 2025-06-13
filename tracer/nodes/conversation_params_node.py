"""Node for generating conversation parameters (number, cost, style) for user profiles."""

from typing import Any

from langchain_core.language_models.base import BaseLanguageModel

# Note: We're not using these prompt functions anymore, but we need the type definitions
from tracer.schemas.graph_state_model import State
from tracer.utils.logging_utils import get_logger

logger = get_logger()

# Type alias for better type safety
VariableData = list[Any] | dict[str, Any]

# Global constants for deterministic parameter generation
DEFAULT_RUNS_NO_VARIABLES = 3
DEFAULT_RUNS_NON_FORWARD_VARIABLES = 3
BASE_COST_PER_CONVERSATION = 0.15
MIN_GOAL_LIMIT = 15
MAX_GOAL_LIMIT = 30
COMBINATIONS_THRESHOLD = 10
MIN_COST = 0.5
MAX_ESTIMATED_RUNS = 10

# --- Helper Functions for extract_profile_variables ---


def _get_profile_variables(profile: dict[str, Any]) -> list[str]:
    """Extracts all defined variable names from a profile."""
    variables = []

    # Check for variables at the top level (original behavior)
    variables.extend(
        [
            var_name
            for var_name, var_def in profile.items()
            if isinstance(var_def, dict) and "function" in var_def and "data" in var_def
        ]
    )

    # Also check for variables nested within the 'goals' list
    if "goals" in profile and isinstance(profile["goals"], list):
        for item in profile["goals"]:
            # If the goal item itself is a dictionary with 'function' and 'data'
            if isinstance(item, dict) and "function" in item and "data" in item:
                # Try to find a name for this variable
                for key in item:
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
                if (
                    var_name in item
                    and isinstance(item[var_name], dict)
                    and "function" in item[var_name]
                    and "data" in item[var_name]
                ):
                    return item[var_name]

    return None


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
                max_size = max(max_size, current_size)
            elif isinstance(data, dict) and all(k in data for k in ["min", "max", "step"]) and data["step"] != 0:
                # Handle range-defined variables
                steps = (data["max"] - data["min"]) / data["step"] + 1
                current_size = int(steps) if steps >= 1 else 1
                max_size = max(max_size, current_size)

    return max_size


def _calculate_variable_size(data: VariableData) -> int:
    """Calculates the size of a single variable based on its data definition.

    Args:
        data: The data definition for the variable.

    Returns:
        The calculated size of the variable.
    """
    if isinstance(data, list):
        return len(data) if data else 1

    if isinstance(data, dict) and all(k in data for k in ["min", "max", "step"]) and data["step"] != 0:
        steps = (data["max"] - data["min"]) / data["step"] + 1
        return int(steps) if steps > 0 else 1

    return 1


def _get_variable_sizes(profile: dict[str, Any], variables: list[str]) -> dict[str, int]:
    """Gets the sizes of all variables in the profile.

    Args:
        profile: The user profile dictionary.
        variables: List of variable names.

    Returns:
        Dictionary mapping variable names to their sizes.
    """
    var_sizes = {}

    for var_name in variables:
        var_def = _get_variable_def(profile, var_name)
        if var_def and "data" in var_def:
            data = var_def.get("data", [])
            var_sizes[var_name] = _calculate_variable_size(data)

    return var_sizes


def _adjust_for_forward_dependencies(profile: dict[str, Any], combinations: int, var_sizes: dict[str, int]) -> int:
    """Adjusts combinations count for forward dependencies to avoid double-counting.

    Args:
        profile: The user profile dictionary.
        combinations: Current combinations count.
        var_sizes: Dictionary of variable sizes.

    Returns:
        Adjusted combinations count.
    """
    if "forward_dependencies" not in profile:
        return combinations

    forward_dependencies = profile["forward_dependencies"]

    for dependent_var, source_vars in forward_dependencies.items():
        if dependent_var not in var_sizes:
            continue

        if any(source_var in var_sizes for source_var in source_vars):
            combinations = combinations // var_sizes[dependent_var]

    return combinations


def _extract_forward_param(func: str) -> str | None:
    """Extracts parameter from forward function string.

    Args:
        func: Function string to parse.

    Returns:
        Extracted parameter or None if not found.
    """
    if not ("forward" in func and "(" in func and ")" in func):
        return None

    param = func.split("(")[1].split(")")[0]
    return param if param and param != "rand" and not param.isdigit() else None


def _adjust_for_nested_forwards(
    variables: list[str], profile: dict[str, Any], combinations: int, var_sizes: dict[str, int]
) -> int:
    """Adjusts combinations for nested forward references in custom format.

    Args:
        variables: List of variable names.
        profile: The user profile dictionary.
        combinations: Current combinations count.
        var_sizes: Dictionary of variable sizes.

    Returns:
        Adjusted combinations count.
    """
    for var_name in variables:
        var_def = _get_variable_def(profile, var_name)
        if not (var_def and "function" in var_def):
            continue

        param = _extract_forward_param(var_def["function"])
        if param and param in var_sizes and var_name in var_sizes:
            combinations = combinations // var_sizes[var_name]

    return combinations


def _calculate_combinations(profile: dict[str, Any], variables: list[str]) -> int:
    """Calculates the potential number of combinations based on variable definitions.

    Args:
        profile: The user profile dictionary.
        variables: List of variable names.

    Returns:
        The calculated number of combinations.
    """
    var_sizes = _get_variable_sizes(profile, variables)
    combinations = 1

    for var_size in var_sizes.values():
        combinations *= var_size

    combinations = _adjust_for_forward_dependencies(profile, combinations, var_sizes)
    combinations = _adjust_for_nested_forwards(variables, profile, combinations, var_sizes)

    return max(combinations, 1)


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


# --- Deterministic Conversation Parameters Generation ---


def _count_outputs(profile: dict[str, Any]) -> int:
    """Count the number of outputs in a profile in a robust way.

    This function handles various possible structures of the output section.
    """
    # First, check for direct 'outputs' key at the top level of the profile
    if "outputs" in profile and isinstance(profile["outputs"], list):
        logger.debug("Found outputs list at top level with %d items", len(profile["outputs"]))
        return len(profile["outputs"])

    # Standard structure check - chatbot.output key path
    if "chatbot" in profile and "output" in profile["chatbot"]:
        output_section = profile["chatbot"]["output"]

        # Handle list structure (most common)
        if isinstance(output_section, list):
            logger.debug("Found output list in chatbot with %d items", len(output_section))
            return len(output_section)

        # Handle dictionary structure (less common)
        if isinstance(output_section, dict):
            logger.debug("Found output dict in chatbot with %d keys", len(output_section))
            return len(output_section)

    # Also check for the plural form 'outputs' inside chatbot
    if "chatbot" in profile and "outputs" in profile["chatbot"]:
        output_section = profile["chatbot"]["outputs"]

        # Handle list structure
        if isinstance(output_section, list):
            logger.debug("Found outputs list in chatbot with %d items", len(output_section))
            return len(output_section)

        # Handle dictionary structure
        if isinstance(output_section, dict):
            logger.debug("Found outputs dict in chatbot with %d keys", len(output_section))
            return len(output_section)

    # None of the expected structures found
    logger.debug("No outputs found in expected locations")
    return 0


def _determine_number_of_conversations(
    max_var_size: int,
    forward_vars: list[str],
    variables: list[str],
    profile: dict[str, Any],
    *,
    has_nested_forwards: bool = False,
) -> str | int:
    """Determines the number of conversations based on profile characteristics.

    Args:
        max_var_size: Maximum size of any variable.
        forward_vars: List of variables with forward dependencies.
        variables: List of all variables.
        profile: The user profile dictionary.
        has_nested_forwards: Whether profile has nested forward dependencies.

    Returns:
        Number value as string or integer.
    """
    if has_nested_forwards:
        total_combinations = _calculate_combinations(profile, variables)
        return "all_combinations" if total_combinations < COMBINATIONS_THRESHOLD else "sample(0.3)"

    if max_var_size > 1:
        return max_var_size

    if forward_vars:
        return DEFAULT_RUNS_NON_FORWARD_VARIABLES

    return DEFAULT_RUNS_NO_VARIABLES


def _calculate_max_cost(number_value: str | int, variables: list[str], profile: dict[str, Any]) -> float:
    """Calculates maximum cost based on number of conversations.

    Args:
        number_value: Number of conversations.
        variables: List of variables.
        profile: The user profile dictionary.

    Returns:
        Calculated maximum cost.
    """
    if isinstance(number_value, int):
        max_cost = BASE_COST_PER_CONVERSATION * number_value
    elif number_value == "all_combinations":
        total_combinations = _calculate_combinations(profile, variables)
        max_cost = BASE_COST_PER_CONVERSATION * min(total_combinations, MAX_ESTIMATED_RUNS)
    elif isinstance(number_value, str) and "sample" in number_value:
        sample_ratio = float(number_value.split("(")[1].split(")")[0])
        total_combinations = _calculate_combinations(profile, variables)
        estimated_runs = round(total_combinations * sample_ratio)
        max_cost = BASE_COST_PER_CONVERSATION * min(estimated_runs, MAX_ESTIMATED_RUNS)
    else:
        max_cost = 1.0

    return max(round(max_cost, 2), MIN_COST)


def _calculate_goal_limit(num_goals: int, num_outputs: int) -> int:
    """Calculates the goal limit based on goals and outputs count.

    Args:
        num_goals: Number of goals.
        num_outputs: Number of outputs.

    Returns:
        Calculated goal limit.
    """
    base_goal_limit = (num_goals + num_outputs) * 2
    return min(max(MIN_GOAL_LIMIT, base_goal_limit), MAX_GOAL_LIMIT)


def _log_profile_structure_debug(profile: dict[str, Any]) -> None:
    """Logs profile structure for debugging purposes.

    Args:
        profile: The user profile dictionary.
    """
    logger.debug("============= PROFILE STRUCTURE DEBUG =============")
    for key, value in profile.items():
        if key == "chatbot" and isinstance(value, dict):
            logger.debug("chatbot:")
            for chat_key, chat_value in value.items():
                if chat_key == "output":
                    logger.debug("  output: (type: %s)", type(chat_value))
                    if isinstance(chat_value, list):
                        for idx, item in enumerate(chat_value):
                            logger.debug("    item %d: %s (type: %s)", idx, item, type(item))
                    elif isinstance(chat_value, dict):
                        for out_key, out_value in chat_value.items():
                            logger.debug("    %s: %s (type: %s)", out_key, out_value, type(out_value))
                else:
                    logger.debug("  %s: %s", chat_key, chat_value)
        else:
            logger.debug("%s: %s", key, type(value))
    logger.debug("=================================================")


def _process_single_profile(profile: dict[str, Any], profile_index: int, total_profiles: int) -> None:
    """Processes a single profile to generate conversation parameters.

    Args:
        profile: The user profile dictionary.
        profile_index: Index of current profile (1-based).
        total_profiles: Total number of profiles.
    """
    profile_name = profile.get("name", f"Profile {profile_index}")

    _log_profile_structure_debug(profile)

    variables, forward_vars, has_nested_forwards, _, _ = extract_profile_variables(profile)
    max_var_size = _get_max_variable_size(profile)

    # Count goals and outputs
    num_goals = 0
    if "goals" in profile and isinstance(profile["goals"], list):
        num_goals = len([g for g in profile["goals"] if isinstance(g, str)])
        logger.debug("Found %d goals in profile", num_goals)

    num_outputs = _count_outputs(profile)
    logger.debug("Counted %d outputs using robust method", num_outputs)
    logger.debug("Final count - Goals: %d, Outputs: %d", num_goals, num_outputs)

    # Generate parameters
    number_value = _determine_number_of_conversations(
        max_var_size, forward_vars, variables, profile, has_nested_forwards=has_nested_forwards
    )
    max_cost = _calculate_max_cost(number_value, variables, profile)
    goal_limit = _calculate_goal_limit(num_goals, num_outputs)

    logger.debug(
        "Goal limit calculation: min(%d, max(%d, (%d goals + %d outputs) * 2)) = %d",
        MAX_GOAL_LIMIT,
        MIN_GOAL_LIMIT,
        num_goals,
        num_outputs,
        goal_limit,
    )

    goal_style = {"steps": goal_limit}
    interaction_styles = ["single question"]

    conversation_params = {
        "number": number_value,
        "max_cost": max_cost,
        "goal_style": goal_style,
        "interaction_style": interaction_styles,
    }

    # Log results
    logger.info(" ✅ Generated conversation parameters %d/%d: '%s'", profile_index, total_profiles, profile_name)

    if variables:
        logger.debug(
            "Parameters: number=%s (from %d variables), cost=%.2f, goal=steps %d, styles: single question",
            number_value,
            len(variables),
            max_cost,
            goal_limit,
        )
    else:
        logger.debug(
            "Parameters: number=%s (no variables), cost=%.2f, goal=steps %d, styles: single question",
            number_value,
            max_cost,
            goal_limit,
        )

    profile["conversation"] = conversation_params


def generate_deterministic_parameters(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generates conversation parameters deterministically without using LLM calls.

    Args:
        profiles: List of user profile dictionaries.

    Returns:
        The list of profiles with an added 'conversation' key containing the generated parameters.
    """
    total_profiles = len(profiles)

    for i, profile in enumerate(profiles, 1):
        _process_single_profile(profile, i, total_profiles)

    return profiles


# --- LangGraph Node ---


def conversation_params_node(state: State, _llm: BaseLanguageModel) -> dict[str, Any]:
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
        profiles_with_params = generate_deterministic_parameters(conversation_goals)

    except Exception:
        logger.exception("Error during parameter generation")
        return {"conversation_goals": conversation_goals}
    else:
        return {"conversation_goals": profiles_with_params}
