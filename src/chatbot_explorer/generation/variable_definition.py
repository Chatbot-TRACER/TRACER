"""Generates structured definitions for variables found in user profile goals."""

import json
from typing import Any, TypedDict

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import VARIABLE_PATTERN
from chatbot_explorer.prompts.variable_definition_prompts import ProfileContext, get_variable_definition_prompt
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_VARIABLES = 3


class VariableDefinitionContext(TypedDict):
    """Context needed to define a single variable.

    Arguments:
        profile: A dictionary containing user profile information.
        goals_text: A string representing the user's goals.
        all_variables: A set of variable names available in the context.
        language_instruction: A string for language model instructions.
        llm: An instance of the language model used for generation.
        max_retries: An integer specifying the maximum number of retry attempts.
    """

    profile: dict[str, Any]
    goals_text: str
    all_variables: set[str]
    language_instruction: str
    llm: BaseLanguageModel
    max_retries: int


def _guess_data_structure(lines: list[str], current_index: int) -> list | dict:
    """Guesses if the DATA section is a list or dict based on the next line."""
    next_line_index = current_index + 1
    if next_line_index < len(lines):
        next_line = lines[next_line_index].strip()
        if ":" in next_line and not next_line.startswith("-"):
            return {}
    return []


def _parse_base_definition(lines: list[str], expected_type: str | None) -> tuple[dict | None, list[str]]:
    """Parses the initial structure (FUNCTION, TYPE, DATA start) from LLM response lines."""
    definition = {}
    data_lines = []
    in_data_section = False

    for i, line in enumerate(lines):
        line_content = line.strip()
        if not line_content:
            continue

        if line_content.startswith("FUNCTION:"):
            definition["function"] = line_content[len("FUNCTION:") :].strip()
            in_data_section = False
        elif line_content.startswith("TYPE:"):
            parsed_type = line_content[len("TYPE:") :].strip()
            definition["type"] = parsed_type
            in_data_section = False
            if expected_type and parsed_type != expected_type:
                logger.warning(
                    "LLM returned type '%s' but expected '%s'. Will use '%s'", parsed_type, expected_type, parsed_type
                )
        elif line_content.startswith("DATA:"):
            in_data_section = True
            definition["data"] = _guess_data_structure(lines, i)
        elif in_data_section:
            data_lines.append(line_content)

    if not definition.get("function") or not definition.get("type") or "data" not in definition:
        return None, []

    return definition, data_lines


def _process_string_data(data_lines: list[str], response_content: str) -> list[str] | None:
    """Processes data lines for a 'string' type variable."""
    processed_data = [
        item_line[2:].strip().strip("'\"")
        for item_line in data_lines
        if item_line.startswith("- ") and item_line[2:].strip().strip("'\"")
    ]
    if not processed_data:
        logger.debug("String variable data is empty. LLM response:\n%s", response_content)
        return None
    return processed_data


def _process_numeric_data(data_lines: list[str], data_type: str) -> dict[str, Any] | None:
    """Processes data lines for 'int' or 'float' type variables."""
    processed_data = {}
    for item_line in data_lines:
        if ":" in item_line:
            try:
                key, value_str = item_line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()
                value = int(value_str) if data_type == "int" else float(value_str)
                processed_data[key] = value
            except ValueError:
                logger.debug("Could not parse numeric value for key '%s': '%s'. Skipping.", key, value_str)

    has_min_max = "min" in processed_data and "max" in processed_data
    has_step_or_linspace = "step" in processed_data or "linspace" in processed_data
    if not (has_min_max and has_step_or_linspace):
        logger.debug("Numeric variable data missing min/max/step or min/max/linspace")
        return None
    return processed_data


def _parse_single_variable_definition(
    response_content: str,
    expected_type: str | None = None,
) -> dict[str, Any] | None:
    """Parses the LLM response for one variable into a structured dictionary."""
    lines = response_content.strip().split("\n")
    definition, data_lines = _parse_base_definition(lines, expected_type)

    if not definition:
        logger.debug("Failed to parse base fields (FUNCTION, TYPE, DATA) from LLM response")
        return None

    data_type = definition.get("type")
    processed_data = None

    if data_type == "string":
        processed_data = _process_string_data(data_lines, response_content)
    elif data_type in ["int", "float"]:
        processed_data = _process_numeric_data(data_lines, data_type)
    else:
        logger.warning("Unknown variable type '%s'. Cannot parse data.", data_type)
        return None

    if processed_data is None:
        return None

    definition["data"] = processed_data
    return definition


def _define_single_variable_with_retry(
    variable_name: str,
    context: VariableDefinitionContext,
) -> dict[str, Any] | None:
    """Attempts to define a single variable using the LLM with retries."""
    logger.debug("Defining variable: '%s'", variable_name)
    other_variables = list(context["all_variables"] - {variable_name})
    parsed_def = None

    profile_context: ProfileContext = {
        "name": context["profile"].get("name", "Unnamed Profile"),
        "role": context["profile"].get("role", "Unknown Role"),
        "goals_text": context["goals_text"],
    }

    for attempt in range(context["max_retries"]):
        prompt = get_variable_definition_prompt(
            profile_context,
            variable_name,
            other_variables,
            context["language_instruction"],
        )
        try:
            response = context["llm"].invoke(prompt)
            parsed_def = _parse_single_variable_definition(response.content)
            if parsed_def:
                logger.debug("Successfully parsed definition for '%s'", variable_name)
                return parsed_def  # Success
            logger.debug("Failed parse attempt %d for '%s'", attempt + 1, variable_name)
        except (ValueError, KeyError, AttributeError):
            logger.exception("Error on attempt %d for '%s'", attempt + 1, variable_name)
            parsed_def = None  # Ensure reset on error

    logger.warning("Failed to define '%s' after %d attempts", variable_name, context["max_retries"])
    return None


def _update_goals_with_definition(goals_list: list[Any], variable_name: str, definition: dict[str, Any]) -> None:
    """Updates or appends the variable definition in the goals list."""
    existing_def_index = -1
    for i, goal_item in enumerate(goals_list):
        if isinstance(goal_item, dict) and variable_name in goal_item:
            existing_def_index = i
            break
    if existing_def_index != -1:
        logger.debug("Updating existing definition for '%s'", variable_name)
        goals_list[existing_def_index] = {variable_name: definition}
    else:
        logger.debug("Adding new definition for '%s'", variable_name)
        goals_list.append({variable_name: definition})


def generate_variable_definitions(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
    functionality_structure: list[dict[str, Any]] = None,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Generates and adds variable definitions to user profile goals using an LLM.

    Iterates through profiles, extracts {{variables}} from string goals,
    prompts the LLM to define each variable's generation method (function, type, data),
    parses the response, and adds the structured definition back into the profile's
    'goals' list as a dictionary. Includes retry logic for LLM calls/parsing.

    Args:
        profiles: A list of profile dictionaries, each expected to have a 'goals' key
                  containing a list of strings and potentially existing definition dicts.
        llm: The language model instance for generating definitions.
        supported_languages: Optional list of languages; the first is used for examples.
        functionality_structure: Optional list of functionality nodes with parameter options.
        max_retries: Maximum attempts to get a valid definition for each variable in case of failure.

    Returns:
        The input list of profiles, modified in-place, where the 'goals' list
        within each profile now includes dictionaries defining the found variables.
    """
    primary_language = ""
    language_instruction = ""
    if supported_languages:
        primary_language = supported_languages[0]
        language_instruction = f"Generate examples/values in {primary_language} where appropriate."

    # Extract parameter options from functionality structure if available
    parameter_options_by_profile = {}
    if functionality_structure:
        for profile in profiles:
            profile_name = profile.get("name", "")
            # Pass the LLM for semantic matching
            parameter_options = _extract_parameter_options_for_profile(
                profile,
                functionality_structure,
                llm,  # Pass the LLM here
            )
            if parameter_options:
                parameter_options_by_profile[profile_name] = parameter_options
                logger.info("Found %d parameter options for profile '%s'", len(parameter_options), profile_name)

    for profile in profiles:
        profile_name = profile.get("name", "Unnamed")
        goals_list = profile.get("goals", [])
        if not isinstance(goals_list, list):
            logger.warning("Profile '%s' has invalid 'goals'. Skipping.", profile_name)
            continue

        string_goals = [goal for goal in goals_list if isinstance(goal, str)]
        all_variables: set[str] = set().union(*(VARIABLE_PATTERN.findall(goal) for goal in string_goals))

        if not all_variables:
            continue

        goals_text = "".join(f"- {goal}\n" for goal in string_goals)

        # Get parameter options for this profile if available
        profile_parameter_options = parameter_options_by_profile.get(profile_name, {})

        # Prepare context dictionary once per profile
        var_def_context: VariableDefinitionContext = {
            "profile": profile,
            "goals_text": goals_text,
            "all_variables": all_variables,
            "language_instruction": language_instruction,
            "llm": llm,
            "max_retries": max_retries,
        }

        for variable_name in sorted(all_variables):
            # Check if we have pre-extracted options for this variable
            if variable_name in profile_parameter_options:
                # Use the extracted options directly
                options = profile_parameter_options[variable_name]
                parsed_def = {
                    "function": "forward()",
                    "type": "string",
                    "data": options,
                }
                logger.debug("Using extracted options for '%s': %s", variable_name, options)
            else:
                # No pre-extracted options, generate with LLM
                parsed_def = _define_single_variable_with_retry(
                    variable_name,
                    var_def_context,
                )
                logger.debug("Generated definition for '%s': %s", variable_name, parsed_def)

            if parsed_def:
                _update_goals_with_definition(goals_list, variable_name, parsed_def)

        logger.verbose(
            "    Generated variables: %d/%d: %s",
            len(all_variables),
            len(all_variables),
            ", ".join(sorted(all_variables)[:3]) + (", ..." if len(all_variables) > MAX_VARIABLES else ""),
        )

    # At the end of generate_variable_definitions:
    for profile in profiles:
        logger.debug("Final variable definitions for '%s':", profile.get("name", "Unnamed"))
        for goal in profile.get("goals", []):
            if isinstance(goal, dict):
                for var_name, var_def in goal.items():
                    if isinstance(var_def, dict) and "data" in var_def:
                        if var_name in parameter_options_by_profile.get(profile.get("name", ""), {}):
                            logger.debug(" ✓ '%s': Using matched parameter options", var_name)
                        else:
                            logger.debug(" ✗ '%s': Using LLM-generated options", var_name)

    return profiles


def _extract_parameter_options_for_profile(
    profile: dict[str, Any],
    functionality_structure: list[dict[str, Any]],
    llm: BaseLanguageModel = None,
) -> dict[str, list[str]]:
    """Extract parameter options from functionality structure for a specific profile.

    Uses direct matching and LLM-based semantic matching to find parameter options.

    Args:
        profile: The profile being processed
        functionality_structure: List of functionality nodes
        llm: Optional language model for semantic matching

    Returns:
        Dictionary mapping variable names to their options
    """
    parameter_options = {}
    profile_name = profile.get("name", "Unnamed")

    # Extract variables from the profile goals
    goal_variables = set()
    for goal in profile.get("goals", []):
        if isinstance(goal, str):
            matches = VARIABLE_PATTERN.findall(goal)
            goal_variables.update(matches)

    if not goal_variables:
        logger.debug("Profile '%s': No variables found in goals", profile_name)
        return parameter_options

    logger.debug(
        "Profile '%s': Found %d variables in goals: %s", profile_name, len(goal_variables), ", ".join(goal_variables)
    )

    # FIRST CHANGE: Extract assigned functionalities for this profile - more flexibly
    # We'll extract both exact functionalities and descriptions for fuzzy matching
    profile_funcs = profile.get("functionalities", [])
    profile_desc = profile.get("role", "").lower()  # Use role for context

    logger.debug(
        "Profile '%s': Looking for parameters matching %d functionalities and description",
        profile_name,
        len(profile_funcs),
    )

    # SECOND CHANGE: Instead of requiring exact matches, let's gather ALL parameters
    # and then match them to our variables
    all_parameters = []

    # Helper function to recursively collect parameters from all nodes
    # without requiring exact functionality name matching
    def collect_all_parameters(node, depth=0):
        node_name = node.get("name", "").lower()
        node_desc = node.get("description", "").lower()

        # Extract all parameters that have options
        for param in node.get("parameters", []):
            if isinstance(param, dict) and param.get("options"):
                # Add node context to the parameter for better matching
                param_with_context = param.copy()
                param_with_context["node_name"] = node.get("name", "")
                param_with_context["node_description"] = node.get("description", "")
                all_parameters.append(param_with_context)

                param_name = param.get("name", "unknown")
                param_options = param.get("options", [])
                logger.debug(
                    "%sFound parameter '%s' in node '%s' with %d options: %s",
                    "  " * depth,
                    param_name,
                    node_name,
                    len(param_options),
                    param_options,
                )

        # Process child nodes
        for child in node.get("children", []):
            collect_all_parameters(child, depth + 1)

    # Collect ALL parameters from the structure
    for node in functionality_structure:
        collect_all_parameters(node)

    logger.debug("Found %d total parameters with options in functionality structure", len(all_parameters))

    # Direct matching of variables to parameters
    for var_name in goal_variables:
        var_lower = var_name.lower()

        # First try exact matches by parameter name
        for param in all_parameters:
            param_name = param.get("name", "").lower()
            param_options = param.get("options", [])

            # Try various matching approaches
            if (
                var_lower == param_name
                or var_lower.replace("_", "") == param_name.replace("_", "")
                or var_lower in param_name
                or param_name in var_lower
            ):
                parameter_options[var_name] = param_options
                logger.debug(
                    "Matched variable '%s' to parameter '%s' with options: %s",
                    var_name,
                    param.get("name", ""),
                    param_options,
                )
                break

    # Use LLM matching for remaining variables
    if llm and goal_variables - set(parameter_options.keys()):
        unmatched_variables = list(goal_variables - set(parameter_options.keys()))
        logger.debug("Using LLM to match %d remaining variables", len(unmatched_variables))
        available_params = []

        for param in all_parameters:
            param_info = {
                "name": param.get("name", ""),
                "description": param.get("description", f"Parameter {param.get('name', '')}"),
                "options": param.get("options", []),
            }
            if param_info["options"]:
                available_params.append(param_info)

        if unmatched_variables and available_params:
            logger.debug(
                "Using LLM to match %d variables to %d parameters", len(unmatched_variables), len(available_params)
            )

            matches = _match_variables_to_parameters_with_llm(unmatched_variables, available_params, llm)

            for var_name, param_info in matches.items():
                parameter_options[var_name] = param_info["options"]
                logger.debug("LLM match: variable '%s' → parameter '%s'", var_name, param_info["name"])

    return parameter_options


def _match_variables_to_parameters_with_llm(
    variables: list[str], parameters: list[dict], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Uses LLM to match variables to parameters based on semantic similarity.

    Args:
        variables: List of variable names to match
        parameters: List of parameter dictionaries (name, description, options)
        llm: Language model for matching

    Returns:
        Dictionary mapping variable names to matched parameter info
    """
    if not variables or not parameters:
        return {}

    # Create the matching prompt
    prompt = f"""
As an expert in semantic matching for conversational systems, match variable names from profile goals with parameter definitions from functionality nodes.

VARIABLES TO MATCH:
{json.dumps(variables, indent=2)}

AVAILABLE PARAMETERS:
{json.dumps(parameters, indent=2)}

IMPORTANT MATCHING RULES:
1. Find the most appropriate parameter for each variable based on name similarity and semantic meaning
2. Match variables ONLY to parameters in the SAME DOMAIN (e.g., don't match drink variables to food parameters)
3. Consider the semantic domain of each variable (e.g., 'drink_type' belongs to beverages domain, 'topping_options' belongs to food additions domain)
4. If a variable name contains specific domain indicators (e.g., 'drink', 'pizza', 'size'), ONLY match with parameters in that domain
5. DO NOT match across incompatible semantic domains under any circumstances
6. If uncertain about domain compatibility, DO NOT create a match

Return your matches in JSON format:
{{
  "variable_name": {{
    "parameter_name": "matched_parameter_name",
    "options": [list, of, parameter, options]
  }},
  ...
}}

Only include matches where you are CERTAIN there is a correct semantic relationship with domain compatibility.
If a variable has no appropriate match, DO NOT include it in the results.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Try to extract JSON from response
        matches = {}
        if "{" in content and "}" in content:
            json_str = content[content.find("{") : content.rfind("}") + 1]
            result = json.loads(json_str)

            # Process the matches
            for var_name, match_info in result.items():
                if var_name in variables:
                    # Find the corresponding parameter
                    param_name = match_info.get("parameter_name")
                    options = match_info.get("options", [])

                    # Use both the original parameter options and any additional options from the LLM
                    for param in parameters:
                        if param.get("name") == param_name:
                            param_options = set(param.get("options", []))
                            # Add any new options from the LLM
                            if options:
                                param_options.update(options)
                            matches[var_name] = {
                                "name": param_name,
                                "options": list(param_options),
                                "description": param.get("description", ""),
                            }
                            logger.debug("Combined options for '%s': %s", var_name, list(param_options))
                            break

        return matches
    except Exception as e:
        logger.warning(f"Error in LLM parameter matching: {e}")
        return {}
