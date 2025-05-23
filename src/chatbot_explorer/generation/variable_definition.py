"""Generates structured definitions for variables found in user profile goals."""

import json
import logging
import re
from collections import defaultdict
from typing import Any, TypedDict

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import VARIABLE_PATTERN, VARIABLE_PATTERNS
from chatbot_explorer.prompts.variable_definition_prompts import (
    ProfileContext,
    get_clean_and_suggest_negative_option_prompt,
    get_variable_definition_prompt,
    get_variable_semantic_validation_prompt,
    get_variable_to_datasource_matching_prompt,
)
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


def _generate_smart_default_options(
    variable_name: str,
    profile_goals_context: str,
    llm: BaseLanguageModel,
    language: str = "English",
) -> dict[str, Any] | None:
    """Generate smart default options for common variable types when no match is found.
    Supports English and Spanish.

    Args:
        variable_name: The variable name without {{ }}
        profile_goals_context: Text containing profile goals for context
        llm: The language model to use
        language: The language to generate options in (defaults to English)

    Returns:
        A dictionary with the variable definition or None if generation fails
    """
    variable_name_lower = variable_name.lower()

    # Determine variable category based on its name using the centralized patterns
    variable_category = None
    for category, patterns in VARIABLE_PATTERNS.items():
        if any(pattern in variable_name_lower for pattern in patterns):
            variable_category = category
            break

    # For _type variables, detect the base concept (service_type → service)
    base_concept = None
    if variable_category == "type":
        for type_marker in VARIABLE_PATTERNS["type"]:
            if type_marker in variable_name_lower:
                parts = variable_name_lower.split(type_marker)
                if parts[0]:
                    base_concept = parts[0].strip("_")
                break

    # Create a language-aware prompt based on the variable category
    prompt = None

    if variable_category == "date":
        prompt = f"""
Generate a list of 4-6 realistic date options in {language} that a user would actually select when scheduling an appointment.
These must be SPECIFIC DATE VALUES, not descriptions or labels.

Examples of GOOD date options:
- "Tomorrow"
- "Monday"
- "Next Friday"
- "December 15"
- "2024-01-20"

Examples of BAD options (do NOT include):
- "The date"
- "Your preferred date"
- "Available dates"
- "Date selection"

The variable is named '{variable_name}' and appears in this context:
{profile_goals_context[:300]}...

Output ONLY a JSON list of strings representing actual selectable dates:
["option1", "option2", "option3", ...]
"""
    elif variable_category == "time":
        prompt = f"""
Generate a list of 4-6 realistic time options in {language} that a user would actually select when scheduling an appointment.
These must be SPECIFIC TIME VALUES, not descriptions or labels.

Examples of GOOD time options:
- "9:00 AM"
- "2:30 PM"
- "14:00"
- "Morning"
- "Afternoon"

Examples of BAD options (do NOT include):
- "The time"
- "Your preferred time"
- "Available times"
- "Time selection"

The variable is named '{variable_name}' and appears in this context:
{profile_goals_context[:300]}...

Output ONLY a JSON list of strings representing actual selectable times:
["option1", "option2", "option3", ...]
"""
    elif variable_category == "type" and base_concept:
        prompt = f"""
Generate a list of 4-6 realistic types/categories of {base_concept} in {language}.
These must be SPECIFIC TYPE NAMES, not descriptions or labels.

Examples of GOOD type options:
- "Basic"
- "Premium"
- "Standard"
- "Express"

Examples of BAD options (do NOT include):
- "The {base_concept} type"
- "Available {base_concept}s"
- "{base_concept.title()} categories"

The variable is named '{variable_name}' and appears in this context:
{profile_goals_context[:300]}...

Output ONLY a JSON list of strings representing actual selectable types:
["option1", "option2", "option3", ...]
"""
    elif variable_category == "number_of":
        prompt = f"""
Generate a list of 4-6 realistic numeric quantities in {language}.
These must be SPECIFIC NUMBERS, not descriptions.

Examples of GOOD quantity options:
- "1"
- "2"
- "5"
- "One"
- "Two"

Examples of BAD options (do NOT include):
- "The quantity"
- "Number of items"
- "Amount needed"

The variable is named '{variable_name}' and appears in this context:
{profile_goals_context[:300]}...

Output ONLY a JSON list of strings representing actual selectable quantities:
["option1", "option2", "option3", ...]
"""
    else:
        # General prompt for any other variable
        prompt = f"""
Generate a list of 4-6 realistic options for a variable named '{variable_name}' in {language}.
These must be ACTUAL VALUES a user would select or input, NOT descriptions or labels.

Examples of GOOD options:
- Specific names, values, or choices
- Concrete options a user can select

Examples of BAD options (do NOT include):
- "The {variable_name}"
- "Your {variable_name}"
- "Available {variable_name}s"
- Descriptions about the variable

The variable appears in this context:
{profile_goals_context[:300]}...

Output ONLY a JSON list of strings representing actual selectable values:
["option1", "option2", "option3", ...]
"""

    if not prompt:
        return None

    try:
        logger.info(f"Generating smart default options for '{variable_name}' in {language}")
        response = llm.invoke(prompt)
        response_text = response.content

        # Extract JSON from the response using a non-greedy approach
        json_match = re.search(r"\[[\s\S]*?\]", response_text)
        if not json_match:
            logger.warning(f"Failed to extract JSON list from smart default response for '{variable_name}'")
            return None

        options_list = json.loads(json_match.group(0))
        if not options_list or not isinstance(options_list, list) or len(options_list) < 2:
            logger.warning(f"Invalid options list generated for '{variable_name}': {options_list}")
            return None

        # Validate the generated options using existing validation
        if not _validate_semantic_match(variable_name, options_list, llm):
            logger.warning(f"Generated options for '{variable_name}' failed validation: {options_list}")
            return None

        # Create the definition
        definition = {
            "function": "forward()",
            "type": "string",
            "data": options_list,
        }
        logger.info(f"Created smart default definition for '{variable_name}' with {len(options_list)} options")
        return definition

    except Exception as e:
        logger.error(f"Error generating smart default options for '{variable_name}': {e}")
        return None


def generate_variable_definitions(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
    functionality_structure: list[dict[str, Any]] = None,
    max_retries: int = 3,
    nested_forward: bool = False,
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
        nested_forward: Whether to create nested forward() chains among variables.

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
            parameter_options = _extract_parameter_options_for_profile(
                profile,
                functionality_structure,
                llm,
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

        # First define all variables with standard forward()
        variable_definitions = {}
        for variable_name in sorted(all_variables):
            # Check if we have pre-extracted options for this variable
            if variable_name in profile_parameter_options:
                # Use the extracted options directly
                dirty_options = profile_parameter_options[variable_name]
                final_options = dirty_options

                logger.debug(
                    f"Variable '{variable_name}': Using pre-matched options. Attempting to clean and get negative suggestion."
                )
                logger.debug(f"  Dirty options for '{variable_name}': {dirty_options}")

                clean_and_negative_prompt = get_clean_and_suggest_negative_option_prompt(
                    dirty_options=dirty_options,
                    variable_name=variable_name,
                    profile_goals_context=goals_text,
                    language=primary_language,
                )

                response_content = llm.invoke(clean_and_negative_prompt).content.strip()

                logger.debug(f"  LLM response for cleaning '{variable_name}': {response_content}")

                cleaned_options = []
                invalid_option = None

                # Parse CLEANED_OPTIONS
                if "CLEANED_OPTIONS:" in response_content:
                    options_section = response_content.split("CLEANED_OPTIONS:")[1].split("INVALID_OPTION_SUGGESTION:")[
                        0
                    ]
                    cleaned_options = [
                        line[2:].strip()
                        for line in options_section.strip().split("\n")
                        if line.strip().startswith("- ") and line[2:].strip()
                    ]
                    # Further unique sort
                    cleaned_options = sorted(list(set(opt for opt in cleaned_options if opt)))

                if not cleaned_options:
                    logger.warning(
                        f"LLM cleaning resulted in no valid options for '{variable_name}'. Original dirty: {dirty_options}. Will try smart defaults or LLM fallback."
                    )
                    # Try smart defaults before general LLM definition
                    parsed_def = _generate_smart_default_options(
                        variable_name=variable_name,
                        profile_goals_context=goals_text,
                        llm=llm,
                        language=primary_language,
                    )
                    if not parsed_def:
                        # Fall back to general LLM definition
                        parsed_def = _define_single_variable_with_retry(
                            variable_name,
                            var_def_context,
                        )
                else:
                    # Use cleaned options as final options
                    final_options = cleaned_options

                    # Parse INVALID_OPTION_SUGGESTION
                    if "INVALID_OPTION_SUGGESTION:" in response_content:
                        invalid_section = response_content.split("INVALID_OPTION_SUGGESTION:")[1].strip()
                        if invalid_section and invalid_section.lower() != "none":
                            invalid_option = invalid_section.split("\n")[0].strip()

                    if not invalid_option:
                        logger.warning(
                            f"LLM did not suggest an invalid option for '{variable_name}'. Original dirty: {dirty_options}"
                        )
                    else:
                        final_options.append(invalid_option)

                    parsed_def = {
                        "function": "forward()",
                        "type": "string",
                        "data": final_options,
                    }
                    logger.debug(
                        f"Using pre-matched options for variable '{variable_name}'. Options count: {len(final_options)}. Preview: {final_options[:3]}{'...' if len(final_options) > 3 else ''}"
                    )
            else:
                # No pre-matched options, try smart defaults first for common patterns
                logger.debug(f"No pre-matched options for '{variable_name}'. Trying smart defaults first.")
                parsed_def = _generate_smart_default_options(
                    variable_name=variable_name, profile_goals_context=goals_text, llm=llm, language=primary_language
                )

                if not parsed_def:
                    # Fall back to general LLM definition
                    logger.debug(f"No smart defaults for '{variable_name}'. Generating definition with LLM.")
                    parsed_def = _define_single_variable_with_retry(
                        variable_name,
                        var_def_context,
                    )

                logger.debug(f"Definition for '{variable_name}': {parsed_def}")

            if parsed_def:
                variable_definitions[variable_name] = parsed_def
                _update_goals_with_definition(goals_list, variable_name, parsed_def)

        # Apply nested forward() chain if requested
        if nested_forward and len(variable_definitions) > 1:
            logger.info(
                f"Creating nested forward() chain for {len(variable_definitions)} variables in profile '{profile_name}'"
            )

            # Sort variable names to ensure deterministic chaining
            sorted_var_names = sorted(variable_definitions.keys())

            # Set up the forward chain
            for i in range(len(sorted_var_names) - 1):
                current_var = sorted_var_names[i]
                next_var = sorted_var_names[i + 1]

                # Update current variable to forward() the next variable
                current_def = variable_definitions[current_var]
                current_def["function"] = f"forward({next_var})"

                # Update the definition in the goals list
                for i, goal_item in enumerate(goals_list):
                    if isinstance(goal_item, dict) and current_var in goal_item:
                        goals_list[i] = {current_var: current_def}
                        break

                logger.debug(f"  Chained variable '{current_var}' to forward('{next_var}')")

            # The last variable keeps its basic forward() function
            logger.debug(f"  Last variable '{sorted_var_names[-1]}' remains with simple forward()")

        logger.verbose(
            "    Generated variables: %d/%d: %s",
            len(all_variables),
            len(all_variables),
            ", ".join(sorted(all_variables)[:3]) + (", ..." if len(all_variables) > MAX_VARIABLES else ""),
        )

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
    all_structured_functionalities: list[dict[str, Any]],
    llm: BaseLanguageModel,
) -> dict[str, list[str]]:
    """Extracts potential option values for goal variables by looking at.

    1. Direct parameter definitions of functionalities assigned to the profile.
    2. Output options of ANY functionality that look like lists of choices.
    3. Uses LLM for semantic matching if direct/output matches are not found or to confirm.
    """
    parameter_options_for_vars: dict[str, set[str]] = defaultdict(set)
    profile_name = profile.get("name", "Unnamed Profile")

    goal_variables = set()
    for goal_str in profile.get("goals", []):
        if isinstance(goal_str, str):
            goal_variables.update(VARIABLE_PATTERN.findall(goal_str))

    if not goal_variables:
        return {}

    logger.debug(f"Profile '{profile_name}': Extracting options for variables: {goal_variables}")

    # --- Prepare a comprehensive list of potential data sources for variables ---
    potential_data_sources = []

    logger.debug("--- Input to _extract_parameter_options_for_profile: Sample Functionalities ---")
    for i, func_d in enumerate(all_structured_functionalities[:15]):  # Log first 15
        logger.debug(
            f"  Func {i}: Name='{func_d.get('name')}', Params='{func_d.get('parameters')}', Outputs='{func_d.get('outputs')}'"
        )

    # Recursive function to process functionality and its children
    def process_functionality(func_dict):
        func_name = func_dict.get("name", "unknown_functionality")

        # Source 1: Direct Parameters of functionality
        for p_dict in func_dict.get("parameters", []):
            if isinstance(p_dict, dict) and p_dict.get("name") and p_dict.get("options"):
                # Only consider parameters that actually have options defined
                if isinstance(p_dict["options"], list) and len(p_dict["options"]) > 0:
                    potential_data_sources.append(
                        {
                            "source_name": p_dict["name"],
                            "source_description": p_dict.get("description")
                            or f"Input options for parameter '{p_dict['name']}' in function '{func_name}'",
                            "options": p_dict["options"],
                            "type": "direct_parameter",
                            "origin_func": func_name,
                        }
                    )

        # Source 2: "Outputs-as-Parameters" from functionality
        for o_dict in func_dict.get("outputs", []):
            if isinstance(o_dict, dict) and o_dict.get("category"):
                output_category = o_dict["category"]
                output_desc = o_dict.get("description", "")

                extracted_options_from_desc = []
                if output_desc:
                    # Look for list-like patterns more intelligently
                    list_patterns = [
                        r"(?:including|such as|like)\s+([^.]+)",  # "including X, Y, Z"
                        r"(?:list of|types of)\s+([^.]+)",  # "list of X, Y, Z"
                        r"\(([^)]+)\)",  # "(X, Y, Z)"
                        r":\s*([^.]+)",  # ": X, Y, Z"
                    ]

                    for pattern in list_patterns:
                        matches = re.findall(pattern, output_desc, re.IGNORECASE)
                        for match in matches:
                            # Split on common delimiters
                            items = re.split(r"[,;]+", match.strip())
                            for item in items:
                                item = item.strip()
                                # Only basic filtering avoid empty items and very long sentences
                                if 2 < len(item) < 100:  # Allow longer phrases but not entire paragraphs
                                    extracted_options_from_desc.append(item)

                # Only add if we found reasonable options (at least 2 good ones)
                if len(extracted_options_from_desc) >= 2:
                    logger.debug(
                        f"  For output '{output_category}' from '{func_name}', extracted options: {extracted_options_from_desc}"
                    )

                    potential_data_sources.append(
                        {
                            "source_name": output_category,
                            "source_description": f"List of choices provided by function '{func_name}' under output category '{output_category}': {output_desc[:100]}...",
                            "options": extracted_options_from_desc[:8],  # Limit to prevent spam
                            "type": "output_as_parameter_options",
                            "origin_func": func_name,
                        }
                    )
                else:
                    logger.debug(
                        f"  For output '{output_category}' from '{func_name}', could not extract meaningful options from description: {output_desc[:100]}..."
                    )

        # Process all children recursively
        for child in func_dict.get("children", []):
            if isinstance(child, dict):
                process_functionality(child)

    # Process all functionalities recursively
    for func_dict in all_structured_functionalities:
        process_functionality(func_dict)

    if not potential_data_sources:
        logger.debug(
            f"Profile '{profile_name}': No potential data sources (parameters with options or list-like outputs) found in any functionalities."
        )
        return {}

    logger.debug(f"Profile '{profile_name}': Found {len(potential_data_sources)} potential data sources for variables.")
    if logger.level <= logging.DEBUG and potential_data_sources:
        for i, src in enumerate(potential_data_sources[:5]):
            logger.debug(
                f"  Potential Source {i}: Name='{src['source_name']}', Type='{src['type']}', Options_Preview='{src['options'][:3]}...' (from func '{src['origin_func']}')"
            )

    # --- Use LLM to match goal_variables to these potential_data_sources ---
    if remaining_unmatched_vars := list(goal_variables):
        logger.info(f"Profile '{profile_name}': Attempting LLM matching for variables: {remaining_unmatched_vars}")

        matched_sources = _match_variables_to_data_sources_with_llm(
            remaining_unmatched_vars, potential_data_sources, llm
        )

        for var_name, source_info in matched_sources.items():
            if source_info and source_info.get("options"):
                parameter_options_for_vars[var_name].update(source_info["options"])
                logger.info(
                    f"  Var '{var_name}': Matched by LLM to source '{source_info.get('source_name')}' (type: {source_info.get('type')}) with options. Assigning options for variable definition."
                )
            else:
                logger.debug(
                    f"  Var '{var_name}': LLM match for source '{source_info.get('source_name')}' did not provide usable options."
                )
    else:
        logger.debug(f"Profile '{profile_name}': All goal variables already covered or no variables to match.")

    # Convert sets to lists for the final output
    final_options_dict = {k: sorted(list(v)) for k, v in parameter_options_for_vars.items()}
    logger.info(
        f"Profile '{profile_name}': Final extracted options for {len(final_options_dict)}/{len(goal_variables)} variables."
    )
    return final_options_dict


def _match_variables_to_data_sources_with_llm(
    goal_variable_names: list[str], potential_data_sources: list[dict[str, Any]], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Match variables to data sources using the LLM to find semantic matches."""
    if not goal_variable_names or not potential_data_sources:
        return {}

    # Match each variable individually for better focus and context
    final_matches = {}
    for variable_name in goal_variable_names:
        final_matches.update(_match_single_variable_to_data_sources(variable_name, potential_data_sources, llm))

    return final_matches


def _match_single_variable_to_data_sources(
    variable_name: str, potential_data_sources: list[dict[str, Any]], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Match a single variable to data sources for better focused matching."""
    # The context should be language-agnostic guidance
    profile_context_for_llm = (
        f"User is interacting with a chatbot regarding '{variable_name}'. "
        f"This variable should contain actual concrete values a user would use, "
        f"not descriptions or explanations about such values."
    )

    prompt = get_variable_to_datasource_matching_prompt(
        [variable_name], potential_data_sources, profile_context_for_llm
    )

    logger.debug(
        f"Attempting LLM matching for variable: {variable_name} against {len(potential_data_sources)} sources."
    )
    logger.debug(f"LLM Matching Prompt for {variable_name}:\n{prompt}")

    response_content = llm.invoke(prompt).content.strip()
    logger.debug(f"LLM Matching Response Content for {variable_name}:\n{response_content}")

    # Parse the simplified response
    if response_content.upper() == "NO_MATCH":
        logger.debug(f"LLM found no appropriate match for variable '{variable_name}'")
        return {}

    # Extract DS ID from response
    ds_match = re.search(r"DS(\d+)", response_content.upper())
    if not ds_match:
        logger.warning(f"Could not parse data source ID from LLM response for '{variable_name}': {response_content}")
        return {}

    try:
        source_index = int(ds_match.group(1)) - 1  # Convert DS3 to index 2
        if 0 <= source_index < len(potential_data_sources):
            matched_source_detail = potential_data_sources[source_index]
            matched_options = matched_source_detail.get("options", [])

            if not matched_options:
                logger.warning(f"Data source DS{ds_match.group(1)} has no options for variable '{variable_name}'")
                return {}

            # Validate that the matched options are semantically appropriate
            is_semantically_valid = _validate_semantic_match(variable_name, matched_options, llm)

            if not is_semantically_valid:
                logger.warning(
                    f"Variable '{variable_name}' matched to source '{matched_source_detail.get('source_name')}' "
                    f"but options failed semantic validation: {matched_options}"
                )
                return {}

            logger.info(
                f"Variable '{variable_name}' successfully matched to '{matched_source_detail.get('source_name')}' "
                f"with {len(matched_options)} options: {matched_options}"
            )

            return {
                variable_name: {
                    "source_name": matched_source_detail.get("source_name"),
                    "type": matched_source_detail.get("type"),
                    "options": matched_options,
                }
            }
        logger.warning(f"Invalid data source index {source_index} for variable '{variable_name}'")
        return {}

    except (ValueError, IndexError) as e:
        logger.warning(f"Error processing data source ID for variable '{variable_name}': {e}")
        return {}


def _validate_semantic_match(variable_name: str, matched_options: list[str], llm: BaseLanguageModel) -> bool:
    """Validate that the matched options are semantically appropriate for the variable name."""
    if not matched_options:
        return False

    # Only minimal pre-filtering to remove obviously broken extractions
    clean_options = []
    for opt in matched_options:
        opt_clean = opt.strip()

        # Only filter out clearly broken extractions (empty, too long, obvious fragments)
        if not opt_clean or len(opt_clean) < 2:
            continue

        # Skip if it's extremely long (likely a full sentence or paragraph)
        if len(opt_clean) > 150:
            continue

        clean_options.append(opt_clean)

    if not clean_options:
        logger.debug(f"All options for '{variable_name}' were filtered out during basic cleaning")
        return False

    # Use LLM for semantic validation - let it make the judgment call
    sample_options = clean_options[:5] if len(clean_options) > 5 else clean_options
    prompt = get_variable_semantic_validation_prompt(variable_name, sample_options)
    response = llm.invoke(prompt).content.strip().lower()

    # Look for clear "yes" or "no" in the response
    has_yes = any(word in response for word in ["yes", "sí", "si"])
    has_no = any(word in response for word in ["no"])

    # Be strict - require clear "yes" without "no"
    is_valid = has_yes and not has_no

    logger.debug(
        f"Semantic validation for '{variable_name}' options {sample_options}: {is_valid}. Response: '{response}'"
    )
    return is_valid
