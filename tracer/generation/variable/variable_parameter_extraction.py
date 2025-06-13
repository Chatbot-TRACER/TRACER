"""Parameter extraction from functionality structures."""

import logging
import re
from collections import defaultdict
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.constants import VARIABLE_PATTERN
from tracer.utils.logging_utils import get_logger

logger = get_logger()

# Constants
MIN_OPTION_LENGTH = 2
MAX_OPTION_LENGTH = 100
MIN_OPTIONS_REQUIRED = 2
MAX_OPTIONS_TO_KEEP = 8
MAX_FUNCTIONALITIES_TO_LOG = 15
MAX_SOURCES_TO_LOG = 5


def extract_parameter_options_for_profile(
    profile: dict[str, Any],
    all_structured_functionalities: list[dict[str, Any]],
    llm: BaseLanguageModel,
) -> dict[str, list[str]]:
    """Extract potential option values for goal variables.

    Looks at:
    1. Direct parameter definitions of functionalities assigned to the profile.
    2. Output options of ANY functionality that look like lists of choices.
    3. Uses LLM for semantic matching if direct/output matches are not found or to confirm.

    Args:
        profile: User profile containing goals with variables
        all_structured_functionalities: List of functionality nodes
        llm: Language model for semantic matching

    Returns:
        Dictionary mapping variable names to lists of options
    """
    parameter_options_for_vars: dict[str, set[str]] = defaultdict(set)
    profile_name = profile.get("name", "Unnamed Profile")

    goal_variables = _extract_goal_variables(profile)
    if not goal_variables:
        return {}

    logger.debug("Profile '%s': Extracting options for variables: %s", profile_name, goal_variables)

    potential_data_sources = _build_potential_data_sources(all_structured_functionalities)
    if not potential_data_sources:
        logger.debug("Profile '%s': No potential data sources found in any functionalities.", profile_name)
        return {}

    _log_data_sources_summary(potential_data_sources, profile_name)

    # Import here to avoid circular imports
    from .variable_matching import match_variables_to_data_sources_with_llm

    # Use LLM to match goal_variables to these potential_data_sources
    if remaining_unmatched_vars := list(goal_variables):
        logger.info("Profile '%s': Attempting LLM matching for variables: %s", profile_name, remaining_unmatched_vars)

        matched_sources = match_variables_to_data_sources_with_llm(
            remaining_unmatched_vars, potential_data_sources, llm
        )

        _process_matched_sources(matched_sources, parameter_options_for_vars)
    else:
        logger.debug("Profile '%s': All goal variables already covered or no variables to match.", profile_name)

    # Convert sets to lists for the final output
    final_options_dict = {k: sorted(v) for k, v in parameter_options_for_vars.items()}
    logger.info(
        "Profile '%s': Final extracted options for %d/%d variables.",
        profile_name,
        len(final_options_dict),
        len(goal_variables),
    )
    return final_options_dict


def _extract_goal_variables(profile: dict[str, Any]) -> set[str]:
    """Extract variables from profile goals."""
    goal_variables = set()
    for goal_str in profile.get("goals", []):
        if isinstance(goal_str, str):
            goal_variables.update(VARIABLE_PATTERN.findall(goal_str))
    return goal_variables


def _build_potential_data_sources(all_structured_functionalities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build comprehensive list of potential data sources for variables."""
    potential_data_sources = []

    logger.debug("--- Input to extract_parameter_options_for_profile: Sample Functionalities ---")
    for i, func_d in enumerate(all_structured_functionalities[:MAX_FUNCTIONALITIES_TO_LOG]):
        logger.debug(
            "  Func %d: Name='%s', Params='%s', Outputs='%s'",
            i,
            func_d.get("name"),
            func_d.get("parameters"),
            func_d.get("outputs"),
        )

    def process_functionality(func_dict: dict[str, Any]) -> None:
        """Process functionality and its children recursively."""
        func_name = func_dict.get("name", "unknown_functionality")

        # Source 1: Direct Parameters of functionality
        _process_direct_parameters(func_dict, func_name, potential_data_sources)

        # Source 2: "Outputs-as-Parameters" from functionality
        _process_output_parameters(func_dict, func_name, potential_data_sources)

        # Process all children recursively
        for child in func_dict.get("children", []):
            if isinstance(child, dict):
                process_functionality(child)

    # Process all functionalities recursively
    for func_dict in all_structured_functionalities:
        process_functionality(func_dict)

    return potential_data_sources


def _process_direct_parameters(
    func_dict: dict[str, Any], func_name: str, potential_data_sources: list[dict[str, Any]]
) -> None:
    """Process direct parameters of functionality."""
    for p_dict in func_dict.get("parameters", []):
        if (
            isinstance(p_dict, dict)
            and p_dict.get("name")
            and p_dict.get("options")
            and isinstance(p_dict["options"], list)
            and len(p_dict["options"]) > 0
        ):
            potential_data_sources.append(  # noqa: PERF401
                {
                    "source_name": p_dict["name"],
                    "source_description": (
                        p_dict.get("description")
                        or f"Input options for parameter '{p_dict['name']}' in function '{func_name}'"
                    ),
                    "options": p_dict["options"],
                    "type": "direct_parameter",
                    "origin_func": func_name,
                }
            )


def _process_output_parameters(
    func_dict: dict[str, Any], func_name: str, potential_data_sources: list[dict[str, Any]]
) -> None:
    """Process output parameters that look like lists of choices."""
    for o_dict in func_dict.get("outputs", []):
        if isinstance(o_dict, dict) and o_dict.get("category"):
            output_category = o_dict["category"]
            output_desc = o_dict.get("description", "")

            extracted_options = _extract_options_from_description(output_desc, output_category, func_name)
            if len(extracted_options) >= MIN_OPTIONS_REQUIRED:
                potential_data_sources.append(
                    {
                        "source_name": output_category,
                        "source_description": (
                            f"List of choices provided by function '{func_name}' "
                            f"under output category '{output_category}': {output_desc[:100]}..."
                        ),
                        "options": extracted_options[:MAX_OPTIONS_TO_KEEP],
                        "type": "output_as_parameter_options",
                        "origin_func": func_name,
                    }
                )


def _extract_options_from_description(output_desc: str, output_category: str, func_name: str) -> list[str]:
    """Extract options from output description using pattern matching."""
    extracted_options_from_desc = []

    if not output_desc:
        return extracted_options_from_desc

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
            for original_item in items:
                cleaned_item = original_item.strip()
                # Only basic filtering avoid empty items and very long sentences
                if MIN_OPTION_LENGTH < len(cleaned_item) < MAX_OPTION_LENGTH:
                    extracted_options_from_desc.append(cleaned_item)

    if len(extracted_options_from_desc) >= MIN_OPTIONS_REQUIRED:
        logger.debug(
            "  For output '%s' from '%s', extracted options: %s",
            output_category,
            func_name,
            extracted_options_from_desc,
        )
    else:
        logger.debug(
            "  For output '%s' from '%s', could not extract meaningful options from description: %s...",
            output_category,
            func_name,
            output_desc[:100],
        )

    return extracted_options_from_desc


def _log_data_sources_summary(potential_data_sources: list[dict[str, Any]], profile_name: str) -> None:
    """Log summary of found data sources."""
    logger.debug(
        "Profile '%s': Found %d potential data sources for variables.", profile_name, len(potential_data_sources)
    )

    if logger.isEnabledFor(logging.DEBUG) and potential_data_sources:
        for i, src in enumerate(potential_data_sources[:MAX_SOURCES_TO_LOG]):
            logger.debug(
                "  Potential Source %d: Name='%s', Type='%s', Options_Preview='%s'... (from func '%s')",
                i,
                src["source_name"],
                src["type"],
                src["options"][:3],
                src["origin_func"],
            )


def _process_matched_sources(matched_sources: dict[str, dict], parameter_options_for_vars: dict[str, set[str]]) -> None:
    """Process matched sources and update parameter options."""
    for var_name, source_info in matched_sources.items():
        if source_info and source_info.get("options"):
            parameter_options_for_vars[var_name].update(source_info["options"])
            logger.info(
                "  Var '%s': Matched by LLM to source '%s' (type: %s) with options. "
                "Assigning options for variable definition.",
                var_name,
                source_info.get("source_name"),
                source_info.get("type"),
            )
        else:
            logger.debug(
                "  Var '%s': LLM match for source '%s' did not provide usable options.",
                var_name,
                source_info.get("source_name") if source_info else "None",
            )
