"""Smart default generation for common variable types."""

import json
import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.constants import VARIABLE_PATTERNS
from tracer.utils.logging_utils import get_logger

logger = get_logger()

# Constants for validation
MIN_OPTIONS_COUNT = 2
MAX_OPTION_LENGTH = 150
MIN_OPTION_LENGTH = 2
CONTEXT_PREVIEW_LENGTH = 300


def generate_smart_default_options(
    variable_name: str,
    profile_goals_context: str,
    llm: BaseLanguageModel,
    language: str = "English",
) -> dict[str, Any] | None:
    """Generate smart default options for common variable types when no match is found.

    Supports English and Spanish. Analyzes variable names to determine their type
    and generates appropriate default options.

    Args:
        variable_name: The variable name without {{ }}
        profile_goals_context: Text containing profile goals for context
        llm: The language model to use
        language: The language to generate options in (defaults to English)

    Returns:
        A dictionary with the variable definition or None if generation fails
    """
    variable_category = _determine_variable_category(variable_name)
    base_concept = _extract_base_concept(variable_name, variable_category)

    prompt = _create_category_prompt(variable_category, variable_name, base_concept, language, profile_goals_context)

    if not prompt:
        return None

    try:
        logger.info("Generating smart default options for '%s' in %s", variable_name, language)
        response = llm.invoke(prompt)
        options_list = _extract_options_from_response(response.content, variable_name)

        if not options_list:
            return None

        # Import here to avoid circular imports
        from .variable_validation import validate_semantic_match

        # Validate the generated options
        if not validate_semantic_match(variable_name, options_list, llm):
            logger.warning("Generated options for '%s' failed validation: %s", variable_name, options_list)
            return None

    except Exception:
        logger.exception("Error generating smart default options for '%s'", variable_name)
        return None
    else:
        # Create the definition
        definition = {
            "function": "forward()",
            "type": "string",
            "data": options_list,
        }
        logger.info("Created smart default definition for '%s' with %d options", variable_name, len(options_list))
        return definition


def _determine_variable_category(variable_name: str) -> str | None:
    """Determine variable category based on its name using centralized patterns."""
    variable_name_lower = variable_name.lower()

    for category, patterns in VARIABLE_PATTERNS.items():
        if any(pattern in variable_name_lower for pattern in patterns):
            return category

    return None


def _extract_base_concept(variable_name: str, variable_category: str) -> str | None:
    """Extract base concept for _type variables (service_type → service)."""
    if variable_category != "type":
        return None

    variable_name_lower = variable_name.lower()
    for type_marker in VARIABLE_PATTERNS["type"]:
        if type_marker in variable_name_lower:
            parts = variable_name_lower.split(type_marker)
            if parts[0]:
                return parts[0].strip("_")

    return None


def _create_category_prompt(
    category: str | None, variable_name: str, base_concept: str | None, language: str, profile_goals_context: str
) -> str | None:
    """Create a language-aware prompt based on the variable category."""
    context_preview = profile_goals_context[:CONTEXT_PREVIEW_LENGTH]

    if category == "date":
        return f"""
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
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable dates:
["option1", "option2", "option3", ...]
"""

    if category == "time":
        return f"""
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
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable times:
["option1", "option2", "option3", ...]
"""

    if category == "type" and base_concept:
        return f"""
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
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable types:
["option1", "option2", "option3", ...]
"""

    if category == "number_of":
        return f"""
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
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable quantities:
["option1", "option2", "option3", ...]
"""

    # General prompt for any other variable
    return f"""
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
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable values:
["option1", "option2", "option3", ...]
"""


def _extract_options_from_response(response_text: str, variable_name: str) -> list[str] | None:
    """Extract and validate options list from LLM response."""
    # Extract JSON from the response using a non-greedy approach
    json_match = re.search(r"\[[\s\S]*?\]", response_text)
    if not json_match:
        logger.warning("Failed to extract JSON list from smart default response for '%s'", variable_name)
        return None

    try:
        options_list = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in smart default response for '%s'", variable_name)
        return None

    if not options_list or not isinstance(options_list, list) or len(options_list) < MIN_OPTIONS_COUNT:
        logger.warning("Invalid options list generated for '%s': %s", variable_name, options_list)
        return None

    return options_list
