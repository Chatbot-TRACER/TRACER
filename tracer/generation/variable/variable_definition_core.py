"""Core variable definition functionality."""

from typing import Any, TypedDict

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.variable_definition_prompts import (
    ProfileContext,
    get_variable_definition_prompt,
)
from tracer.utils.logging_utils import get_logger

logger = get_logger()


class VariableDefinitionContext(TypedDict):
    """Context needed to define a single variable.

    Arguments:
        profile: A dictionary containing user profile information.
        goals_text: A string representing the user's goals.
        all_variables: A set of variable names available in the context.
        language_instruction: A string for language model instructions.
        primary_language: The primary language for generating examples/values.
        llm: An instance of the language model used for generation.
        max_retries: An integer specifying the maximum number of retry attempts.
    """

    profile: dict[str, Any]
    goals_text: str
    all_variables: set[str]
    language_instruction: str
    primary_language: str
    llm: BaseLanguageModel
    max_retries: int


def define_single_variable_with_retry(
    variable_name: str,
    context: VariableDefinitionContext,
) -> dict[str, Any] | None:
    """Attempt to define a single variable using the LLM with retries.

    Args:
        variable_name: Name of the variable to define
        context: Context containing profile and LLM information

    Returns:
        Variable definition dictionary or None if failed
    """
    logger.debug("Defining variable: '%s'", variable_name)
    other_variables = list(context["all_variables"] - {variable_name})

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

    logger.warning("Failed to define '%s' after %d attempts", variable_name, context["max_retries"])
    return None


def update_goals_with_definition(goals_list: list[Any], variable_name: str, definition: dict[str, Any]) -> None:
    """Update or append the variable definition in the goals list.

    Args:
        goals_list: List of goals to update
        variable_name: Name of the variable
        definition: Variable definition to add/update
    """
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


def _guess_data_structure(lines: list[str], current_index: int) -> list | dict:
    """Guess if the DATA section is a list or dict based on the next line."""
    next_line_index = current_index + 1
    if next_line_index < len(lines):
        next_line = lines[next_line_index].strip()
        if ":" in next_line and not next_line.startswith("-"):
            return {}
    return []


def _parse_base_definition(lines: list[str], expected_type: str | None) -> tuple[dict | None, list[str]]:
    """Parse the initial structure (FUNCTION, TYPE, DATA start) from LLM response lines."""
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
    """Process data lines for a 'string' type variable."""
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
    """Process data lines for 'int' or 'float' type variables."""
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
    """Parse the LLM response for one variable into a structured dictionary."""
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
