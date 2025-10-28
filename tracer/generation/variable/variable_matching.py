"""Variable to data source matching using LLM."""

import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.variable_definition_prompts import get_variable_to_datasource_matching_prompt
from tracer.utils.logging_utils import get_logger

from .variable_validation import validate_semantic_match

logger = get_logger()


def match_variables_to_data_sources_with_llm(
    goal_variable_names: list[str], potential_data_sources: list[dict[str, Any]], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Match variables to data sources using the LLM to find semantic matches.

    Args:
        goal_variable_names: List of variable names to match
        potential_data_sources: List of potential data sources with options
        llm: Language model for matching

    Returns:
        Dictionary mapping variable names to matched source information
    """
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
        "Attempting LLM matching for variable: %s against %d sources.", variable_name, len(potential_data_sources)
    )
    logger.debug("LLM Matching Prompt for %s:\n%s", variable_name, prompt)

    response_content = llm.invoke(prompt).content.strip()
    logger.debug("LLM Matching Response Content for %s:\n%s", variable_name, response_content)

    return _parse_matching_response(response_content, variable_name, potential_data_sources, llm)


def _parse_matching_response(
    response_content: str, variable_name: str, potential_data_sources: list[dict[str, Any]], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Parse the LLM response for variable matching."""
    # Parse the simplified response
    if response_content.upper() == "NO_MATCH":
        logger.debug("LLM found no appropriate match for variable '%s'", variable_name)
        return {}

    # Extract DS ID from response
    ds_match = re.search(r"DS(\d+)", response_content.upper())
    if not ds_match:
        logger.warning("Could not parse data source ID from LLM response for '%s': %s", variable_name, response_content)
        return {}

    try:
        source_index = int(ds_match.group(1)) - 1  # Convert DS3 to index 2
        if 0 <= source_index < len(potential_data_sources):
            return _process_matched_source(variable_name, source_index, potential_data_sources, llm)

    except (ValueError, IndexError):
        logger.exception("Error processing data source ID for variable '%s'", variable_name)

    logger.warning("Invalid data source index for variable '%s'", variable_name)
    return {}


def _process_matched_source(
    variable_name: str, source_index: int, potential_data_sources: list[dict[str, Any]], llm: BaseLanguageModel
) -> dict[str, dict]:
    """Process a matched data source and validate it."""
    matched_source_detail = potential_data_sources[source_index]
    matched_options = matched_source_detail.get("options", [])

    if not matched_options:
        logger.warning("Data source DS%d has no options for variable '%s'", source_index + 1, variable_name)
        return {}

    # Validate that the matched options are semantically appropriate
    is_semantically_valid = validate_semantic_match(variable_name, matched_options, llm)

    if not is_semantically_valid:
        logger.warning(
            "Variable '%s' matched to source '%s' but options failed semantic validation: %s",
            variable_name,
            matched_source_detail.get("source_name"),
            matched_options,
        )
        return {}

    logger.info(
        "Variable '%s' successfully matched to '%s' with %d options: %s",
        variable_name,
        matched_source_detail.get("source_name"),
        len(matched_options),
        matched_options,
    )

    return {
        variable_name: {
            "source_name": matched_source_detail.get("source_name"),
            "type": matched_source_detail.get("type"),
            "options": matched_options,
        }
    }
