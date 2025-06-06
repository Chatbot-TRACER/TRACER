"""Prompts for generating conversation parameters based on user profile information.

Note: These prompt functions are no longer used since we've replaced LLM-based parameter
generation with deterministic generation in conversation_params_node.py.

The TypedDict classes are kept for backward compatibility with any code that might
reference these types.
"""

from typing import Any, TypedDict

# --- Data Structures for Prompt Arguments ---


class PromptProfileContext(TypedDict):
    """Context related to the user profile for prompts.

    Args:
        profile: The user profile dictionary.
        variables_info: Descriptive string about profile variables.
        language_info: Descriptive string about supported languages.
    """

    profile: dict[str, Any]
    variables_info: str
    language_info: str


class PromptPreviousParams(TypedDict):
    """Previously determined parameters for prompts.

    Args:
        number_value: The determined 'number' parameter (int or str).
        max_cost: The determined 'max_cost' parameter (float).
        goal_style: The determined 'goal_style' dictionary.
    """

    number_value: int | str
    max_cost: float
    goal_style: dict[str, Any]


class PromptLanguageSupport(TypedDict):
    """Language support information for prompts.

    Args:
        supported_languages_text: String listing supported languages (e.g., "English, Spanish").
        languages_example: Formatted string showing language examples for the prompt.
    """

    supported_languages_text: str
    languages_example: str


# Note: All prompt functions have been removed as they are no longer used.
# The parameter generation now uses a deterministic approach defined in
# conversation_params_node.py.
