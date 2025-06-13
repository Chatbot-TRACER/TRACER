"""Semantic validation for variable options."""

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.variable_definition_prompts import get_variable_semantic_validation_prompt
from tracer.utils.logging_utils import get_logger

logger = get_logger()

# Constants for validation
MIN_OPTION_LENGTH = 2
MAX_OPTION_LENGTH = 150
MIN_CLEAN_OPTIONS = 2
SAMPLE_SIZE = 5
RESPONSE_CHECK_LINES = 3


def validate_semantic_match(variable_name: str, matched_options: list[str], llm: BaseLanguageModel) -> bool:
    """Validate that the matched options are semantically appropriate for the variable name.

    Args:
        variable_name: The name of the variable to validate
        matched_options: List of options to validate
        llm: Language model for validation

    Returns:
        True if options are semantically valid, False otherwise
    """
    if not matched_options:
        return False

    clean_options = _clean_options(matched_options, variable_name)
    if not clean_options:
        return False

    return _validate_with_llm(variable_name, clean_options, llm)


def _clean_options(matched_options: list[str], variable_name: str) -> list[str]:
    """Clean options by removing obviously broken extractions."""
    clean_options = []

    for opt in matched_options:
        opt_clean = opt.strip()

        # Filter out clearly broken extractions (empty, too long, obvious fragments)
        if not opt_clean or len(opt_clean) < MIN_OPTION_LENGTH:
            continue

        # Skip if it's extremely long (likely a full sentence or paragraph)
        if len(opt_clean) > MAX_OPTION_LENGTH:
            continue

        clean_options.append(opt_clean)

    if not clean_options:
        logger.debug("All options for '%s' were filtered out during basic cleaning", variable_name)

    return clean_options


def _validate_with_llm(variable_name: str, clean_options: list[str], llm: BaseLanguageModel) -> bool:
    """Use LLM for semantic validation with chain of thought."""
    # Use LLM for semantic validation with chain of thought
    sample_options = clean_options[:SAMPLE_SIZE] if len(clean_options) > SAMPLE_SIZE else clean_options
    prompt = get_variable_semantic_validation_prompt(variable_name, sample_options)

    try:
        response = llm.invoke(prompt).content.strip()
        final_answer = _parse_validation_response(response, variable_name)

        is_valid = final_answer == "yes"
    except Exception:
        logger.exception("Error in semantic validation for '%s'", variable_name)
        # Fall back to being permissive if LLM fails
        return len(clean_options) >= MIN_CLEAN_OPTIONS
    else:
        logger.debug(
            "Semantic validation for '%s' options %s: %s. Final answer: '%s'. Full response: '%s'...",
            variable_name,
            sample_options,
            is_valid,
            final_answer,
            response[:100],
        )
        return is_valid


def _parse_validation_response(response: str, variable_name: str) -> str:
    """Parse the final answer from the chain of thought response."""
    lines = response.split("\n")
    final_answer = None

    # Check the last few lines for a clear Yes/No answer
    for line in reversed(lines[-RESPONSE_CHECK_LINES:]):
        line_clean = line.strip().lower()
        if line_clean in {"yes", "no"}:
            final_answer = line_clean
            break
        # Also check for lines that end with yes/no
        if line_clean.endswith((" yes", " no")):
            final_answer = line_clean.split()[-1]
            break

    if final_answer is None:
        # Fallback: look for yes/no anywhere in the response
        response_lower = response.lower()
        has_yes = "yes" in response_lower
        has_no = "no" in response_lower

        # If we have both or neither, log a warning
        if (has_yes and has_no) or (not has_yes and not has_no):
            logger.warning("Ambiguous validation response for '%s': %s...", variable_name, response[:100])
            return "no"  # Conservative approach

        final_answer = "yes" if has_yes else "no"

    return final_answer
