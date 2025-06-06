"""Node for generating conversation parameters (number, cost, style) for user profiles."""

from typing import Any

from langchain_core.language_models.base import BaseLanguageModel

# Note: We're not using these prompt functions anymore, but we need the type definitions
from tracer.schemas.graph_state_model import State
from tracer.utils.logging_utils import get_logger

logger = get_logger()

# Global constants for deterministic parameter generation
DEFAULT_RUNS_NO_VARIABLES = 3
DEFAULT_RUNS_NON_FORWARD_VARIABLES = 3
BASE_COST_PER_CONVERSATION = 0.15
MIN_GOAL_LIMIT = 15
MAX_GOAL_LIMIT = 30

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


# --- Deterministic Conversation Parameters Generation ---


def _count_outputs(profile: dict[str, Any]) -> int:
    """Count the number of outputs in a profile in a robust way.

    This function handles various possible structures of the output section.
    """
    # First, check for direct 'outputs' key at the top level of the profile
    if "outputs" in profile and isinstance(profile["outputs"], list):
        logger.debug(f"Found outputs list at top level with {len(profile['outputs'])} items")
        return len(profile["outputs"])

    # Standard structure check - chatbot.output key path
    if "chatbot" in profile and "output" in profile["chatbot"]:
        output_section = profile["chatbot"]["output"]

        # Handle list structure (most common)
        if isinstance(output_section, list):
            logger.debug(f"Found output list in chatbot with {len(output_section)} items")
            return len(output_section)

        # Handle dictionary structure (less common)
        if isinstance(output_section, dict):
            logger.debug(f"Found output dict in chatbot with {len(output_section)} keys")
            return len(output_section)

    # Also check for the plural form 'outputs' inside chatbot
    if "chatbot" in profile and "outputs" in profile["chatbot"]:
        output_section = profile["chatbot"]["outputs"]

        # Handle list structure
        if isinstance(output_section, list):
            logger.debug(f"Found outputs list in chatbot with {len(output_section)} items")
            return len(output_section)

        # Handle dictionary structure
        if isinstance(output_section, dict):
            logger.debug(f"Found outputs dict in chatbot with {len(output_section)} keys")
            return len(output_section)

    # None of the expected structures found
    logger.debug("No outputs found in expected locations")
    return 0


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

        # DIRECT STRUCTURE DEBUGGING - Print the raw dict structure
        logger.debug("============= PROFILE STRUCTURE DEBUG =============")
        for key, value in profile.items():
            if key == "chatbot" and isinstance(value, dict):
                logger.debug("chatbot:")
                for chat_key, chat_value in value.items():
                    if chat_key == "output":
                        logger.debug(f"  output: (type: {type(chat_value)})")
                        if isinstance(chat_value, list):
                            for idx, item in enumerate(chat_value):
                                logger.debug(f"    item {idx}: {item} (type: {type(item)})")
                        elif isinstance(chat_value, dict):
                            for out_key, out_value in chat_value.items():
                                logger.debug(f"    {out_key}: {out_value} (type: {type(out_value)})")
                    else:
                        logger.debug(f"  {chat_key}: {chat_value}")
            else:
                logger.debug(f"{key}: {type(value)}")
        logger.debug("=================================================")

        # Extract variables information
        variables, forward_vars, has_nested_forwards, _, _ = extract_profile_variables(profile)

        # Get the maximum variable size
        max_var_size = _get_max_variable_size(profile)

        # Count goals for limit calculation
        num_goals = 0
        if "goals" in profile and isinstance(profile["goals"], list):
            num_goals = len([g for g in profile["goals"] if isinstance(g, str)])
            logger.debug(f"Found {num_goals} goals in profile")

        # Use simple robust counting function
        num_outputs = _count_outputs(profile)
        logger.debug(f"Counted {num_outputs} outputs using robust method")

        logger.debug(f"Final count - Goals: {num_goals}, Outputs: {num_outputs}")

        # Determine number of conversations
        if has_nested_forwards:
            total_combinations = _calculate_combinations(profile, variables)
            if total_combinations < 10:
                number_value = "all_combinations"
            else:
                number_value = "sample(0.3)"
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

        # Always use steps instead of all_answered since all_answered may finish the execution earlier
        base_goal_limit = (num_goals + num_outputs) * 2
        min_limit = MIN_GOAL_LIMIT
        max_limit = MAX_GOAL_LIMIT
        goal_limit = min(max(min_limit, base_goal_limit), max_limit)

        logger.debug(
            f"Goal limit calculation: min({max_limit}, max({min_limit}, ({num_goals} goals + {num_outputs} outputs) * 2)) = {goal_limit}"
        )

        goal_style = {"steps": goal_limit}

        # Always use "single question" as interaction style
        interaction_styles = ["single question"]

        # Build the conversation parameters
        conversation_params = {
            "number": number_value,
            "max_cost": max_cost,
            "goal_style": goal_style,
            "interaction_style": interaction_styles,
        }

        # Log what we've generated
        style_summary = ", styles: single question"

        logger.info(" ✅ Generated conversation parameters %d/%d: '%s'", i, total_profiles, profile_name)

        if variables:
            logger.debug(
                "Parameters: number=%s (from %d variables), cost=%.2f, goal=steps %d %s",
                number_value,
                len(variables),
                max_cost,
                goal_limit,
                style_summary,
            )
        else:
            logger.debug(
                "Parameters: number=%s (no variables), cost=%.2f, goal=steps %d %s",
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
