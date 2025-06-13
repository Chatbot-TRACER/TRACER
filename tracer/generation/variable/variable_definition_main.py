"""Main orchestration for variable definition generation."""

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.constants import VARIABLE_PATTERN
from tracer.prompts.variable_definition_prompts import (
    get_clean_and_suggest_negative_option_prompt,
)
from tracer.utils.logging_utils import get_logger

from .variable_definition_core import VariableDefinitionContext
from .variable_parameter_extraction import extract_parameter_options_for_profile
from .variable_smart_defaults import generate_smart_default_options

logger = get_logger()

# Constants
MAX_VARIABLES_TO_SHOW = 3


@dataclass
class VariableDefinitionConfig:
    """Configuration for variable definition generation."""

    supported_languages: list[str] | None = None
    functionality_structure: list[dict[str, Any]] | None = None
    max_retries: int = 3
    nested_forward: bool = False


def generate_variable_definitions(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    *,
    config: VariableDefinitionConfig | None = None,
) -> list[dict[str, Any]]:
    """Generate and add variable definitions to user profile goals using an LLM.

    Iterates through profiles, extracts {{variables}} from string goals,
    prompts the LLM to define each variable's generation method (function, type, data),
    parses the response, and adds the structured definition back into the profile's
    'goals' list as a dictionary. Includes retry logic for LLM calls/parsing.

    Args:
        profiles: A list of profile dictionaries, each expected to have a 'goals' key
                  containing a list of strings and potentially existing definition dicts.
        llm: The language model instance for generating definitions.
        config: Optional configuration object to override default settings.

    Returns:
        The input list of profiles, modified in-place, where the 'goals' list
        within each profile now includes dictionaries defining the found variables.
    """
    if config is None:
        config = VariableDefinitionConfig()

    primary_language, language_instruction = _prepare_language_settings(config.supported_languages)
    parameter_options_by_profile = _extract_parameter_options_by_profile(profiles, config.functionality_structure, llm)

    for profile in profiles:
        profile_context = ProfileProcessingContext(
            profile=profile,
            llm=llm,
            primary_language=primary_language,
            language_instruction=language_instruction,
            parameter_options_by_profile=parameter_options_by_profile,
            config=config,
        )
        _process_single_profile(profile_context)

    _log_final_summary(profiles, parameter_options_by_profile)
    return profiles


def _prepare_language_settings(supported_languages: list[str] | None) -> tuple[str, str]:
    """Prepare language settings for variable generation."""
    primary_language = ""
    language_instruction = ""

    if supported_languages:
        primary_language = supported_languages[0]
        language_instruction = f"Generate examples/values in {primary_language} where appropriate."

    return primary_language, language_instruction


def _extract_parameter_options_by_profile(
    profiles: list[dict[str, Any]], functionality_structure: list[dict[str, Any]] | None, llm: BaseLanguageModel
) -> dict[str, dict[str, list[str]]]:
    """Extract parameter options from functionality structure if available."""
    parameter_options_by_profile = {}

    if not functionality_structure:
        return parameter_options_by_profile

    for profile in profiles:
        profile_name = profile.get("name", "")
        parameter_options = extract_parameter_options_for_profile(profile, functionality_structure, llm)

        if parameter_options:
            parameter_options_by_profile[profile_name] = parameter_options
            logger.info("Found %d parameter options for profile '%s'", len(parameter_options), profile_name)

    return parameter_options_by_profile


@dataclass
class ProfileProcessingContext:
    """Context for processing a single profile."""

    profile: dict[str, Any]
    llm: BaseLanguageModel
    primary_language: str
    language_instruction: str
    parameter_options_by_profile: dict[str, dict[str, list[str]]]
    config: VariableDefinitionConfig


def _process_single_profile(context: ProfileProcessingContext) -> None:
    """Process a single profile to generate variable definitions."""
    profile_name = context.profile.get("name", "Unnamed")
    goals_list = context.profile.get("goals", [])

    if not isinstance(goals_list, list):
        logger.warning("Profile '%s' has invalid 'goals'. Skipping.", profile_name)
        return

    all_variables = _extract_variables_from_goals(goals_list)
    if not all_variables:
        return

    goals_text = "".join(f"- {goal}\n" for goal in goals_list if isinstance(goal, str))
    profile_parameter_options = context.parameter_options_by_profile.get(profile_name, {})

    # Prepare context dictionary once per profile
    var_def_context = VariableDefinitionContext(
        profile=context.profile,
        goals_text=goals_text,
        all_variables=all_variables,
        language_instruction=context.language_instruction,
        primary_language=context.primary_language,
        llm=context.llm,
        max_retries=context.config.max_retries,
    )

    # Define all variables
    variable_definitions = _define_all_variables(all_variables, profile_parameter_options, var_def_context, goals_list)

    # Apply nested forward() chain if requested
    if context.config.nested_forward and len(variable_definitions) > 1:
        _apply_nested_forward_chain(variable_definitions, goals_list, profile_name)

    _log_profile_summary(all_variables)


def _extract_variables_from_goals(goals_list: list) -> set[str]:
    """Extract variables from string goals."""
    string_goals = [goal for goal in goals_list if isinstance(goal, str)]
    return set().union(*(VARIABLE_PATTERN.findall(goal) for goal in string_goals))


def _define_all_variables(
    all_variables: set[str],
    profile_parameter_options: dict[str, list[str]],
    var_def_context: VariableDefinitionContext,
    goals_list: list,
) -> dict[str, dict[str, Any]]:
    """Define all variables for a profile."""
    variable_definitions = {}

    # Extract values from context
    goals_text = var_def_context["goals_text"]
    llm = var_def_context["llm"]
    primary_language = var_def_context["primary_language"]

    # Import here to avoid circular imports
    from .variable_definition_core import (
        update_goals_with_definition,
    )

    for variable_name in sorted(all_variables):
        parsed_def = None

        if variable_name in profile_parameter_options:
            # Use pre-extracted options
            parsed_def = _process_pre_extracted_options(
                variable_name, profile_parameter_options[variable_name], goals_text, primary_language, llm
            )
        else:
            # Try smart defaults first, then LLM fallback
            parsed_def = _process_variable_without_options(
                variable_name, goals_text, primary_language, llm, var_def_context
            )

        if parsed_def:
            variable_definitions[variable_name] = parsed_def
            update_goals_with_definition(goals_list, variable_name, parsed_def)

    return variable_definitions


def _process_pre_extracted_options(
    variable_name: str, dirty_options: list[str], goals_text: str, primary_language: str, llm: BaseLanguageModel
) -> dict[str, Any] | None:
    """Process variables that have pre-extracted options."""
    logger.debug(
        "Variable '%s': Using pre-matched options. Attempting to clean and get negative suggestion.", variable_name
    )
    logger.debug("  Dirty options for '%s': %s", variable_name, dirty_options)

    clean_and_negative_prompt = get_clean_and_suggest_negative_option_prompt(
        dirty_options=dirty_options,
        variable_name=variable_name,
        profile_goals_context=goals_text,
        language=primary_language,
    )

    response_content = llm.invoke(clean_and_negative_prompt).content.strip()
    logger.debug("  LLM response for cleaning '%s': %s", variable_name, response_content)

    cleaned_options, invalid_option = _parse_cleaning_response(response_content)

    if not cleaned_options:
        logger.warning(
            "LLM cleaning resulted in no valid options for '%s'. Original dirty: %s. Will try smart defaults or LLM fallback.",
            variable_name,
            dirty_options,
        )
        return generate_smart_default_options(variable_name, goals_text, llm, primary_language)

    # Use cleaned options as final options
    final_options = cleaned_options
    if invalid_option:
        final_options.append(invalid_option)
    else:
        logger.warning(
            "LLM did not suggest an invalid option for '%s'. Original dirty: %s", variable_name, dirty_options
        )

    parsed_def = {
        "function": "forward()",
        "type": "string",
        "data": final_options,
    }

    preview_count = 3
    preview = final_options[:preview_count]
    ellipsis = "..." if len(final_options) > preview_count else ""
    logger.debug(
        "Using pre-matched options for variable '%s'. Options count: %d. Preview: %s%s",
        variable_name,
        len(final_options),
        preview,
        ellipsis,
    )

    return parsed_def


def _parse_cleaning_response(response_content: str) -> tuple[list[str], str | None]:
    """Parse the LLM response for option cleaning."""
    cleaned_options = []
    invalid_option = None

    # Parse CLEANED_OPTIONS
    if "CLEANED_OPTIONS:" in response_content:
        options_section = response_content.split("CLEANED_OPTIONS:")[1].split("INVALID_OPTION_SUGGESTION:")[0]
        cleaned_options = [
            line[2:].strip()
            for line in options_section.strip().split("\n")
            if line.strip().startswith("- ") and line[2:].strip()
        ]
        # Further unique sort
        cleaned_options = sorted({opt for opt in cleaned_options if opt})

    # Parse INVALID_OPTION_SUGGESTION
    if "INVALID_OPTION_SUGGESTION:" in response_content:
        invalid_section = response_content.split("INVALID_OPTION_SUGGESTION:")[1].strip()
        if invalid_section and invalid_section.lower() != "none":
            invalid_option = invalid_section.split("\n")[0].strip()

    return cleaned_options, invalid_option


def _process_variable_without_options(
    variable_name: str,
    goals_text: str,
    primary_language: str,
    llm: BaseLanguageModel,
    var_def_context: VariableDefinitionContext,
) -> dict[str, Any] | None:
    """Process variables that don't have pre-extracted options."""
    logger.debug("No pre-matched options for '%s'. Trying smart defaults first.", variable_name)

    # Try smart defaults first
    parsed_def = generate_smart_default_options(variable_name, goals_text, llm, primary_language)

    if not parsed_def:
        # Fall back to general LLM definition
        logger.debug("No smart defaults for '%s'. Generating definition with LLM.", variable_name)

        # Import here to avoid circular imports
        from .variable_definition_core import define_single_variable_with_retry

        parsed_def = define_single_variable_with_retry(variable_name, var_def_context)

    logger.debug("Definition for '%s': %s", variable_name, parsed_def)
    return parsed_def


def _apply_nested_forward_chain(
    variable_definitions: dict[str, dict[str, Any]], goals_list: list, profile_name: str
) -> None:
    """Apply nested forward() chain among variables."""
    logger.info(
        "Creating nested forward() chain for %d variables in profile '%s'", len(variable_definitions), profile_name
    )

    # Sort variable names to ensure deterministic chaining
    sorted_var_names = sorted(variable_definitions.keys())

    # Import here to avoid circular imports

    # Set up the forward chain
    for i in range(len(sorted_var_names) - 1):
        current_var = sorted_var_names[i]
        next_var = sorted_var_names[i + 1]

        # Update current variable to forward() the next variable
        current_def = variable_definitions[current_var]
        current_def["function"] = f"forward({next_var})"

        # Update the definition in the goals list
        for j, goal_item in enumerate(goals_list):
            if isinstance(goal_item, dict) and current_var in goal_item:
                goals_list[j] = {current_var: current_def}
                break

        logger.debug("  Chained variable '%s' to forward('%s')", current_var, next_var)

    # The last variable keeps its basic forward() function
    logger.debug("  Last variable '%s' remains with simple forward()", sorted_var_names[-1])


def _log_profile_summary(all_variables: set[str]) -> None:
    """Log summary for a processed profile."""
    variable_list = sorted(all_variables)
    displayed_vars = variable_list[:MAX_VARIABLES_TO_SHOW]
    ellipsis = ", ..." if len(variable_list) > MAX_VARIABLES_TO_SHOW else ""

    logger.verbose(
        "    Generated variables: %d/%d: %s%s",
        len(all_variables),
        len(all_variables),
        ", ".join(displayed_vars),
        ellipsis,
    )


def _log_final_summary(
    profiles: list[dict[str, Any]], parameter_options_by_profile: dict[str, dict[str, list[str]]]
) -> None:
    """Log final summary of variable definitions."""
    for profile in profiles:
        profile_name = profile.get("name", "Unnamed")
        logger.debug("Final variable definitions for '%s':", profile_name)

        for goal in profile.get("goals", []):
            if isinstance(goal, dict):
                for var_name, var_def in goal.items():
                    if isinstance(var_def, dict) and "data" in var_def:
                        if var_name in parameter_options_by_profile.get(profile_name, {}):
                            logger.debug(" ✓ '%s': Using matched parameter options", var_name)
                        else:
                            logger.debug(" ✗ '%s': Using LLM-generated options", var_name)
