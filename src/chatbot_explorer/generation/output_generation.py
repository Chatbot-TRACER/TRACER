"""Generates structured output definitions based on user profiles and functionalities."""

import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import VARIABLE_PATTERN
from chatbot_explorer.prompts.output_generation_prompts import get_outputs_prompt
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()


def _parse_llm_outputs(llm_content: str) -> list[dict[str, Any]]:
    """Parses the LLM's raw text output into a structured list of output definitions.

    Expects input lines like:
        OUTPUT: output_name
        TYPE: data_type
        DESCRIPTION: Some description

    Args:
        llm_content: The raw string content from the LLM response.

    Returns:
        A list of dictionaries, where each dictionary represents an output field
        with its name, type, and description. Example:
        [{'output_name': {'type': 'str', 'description': '...'}}]
    """
    outputs_list = []
    current_output_name = None
    current_data = {}

    for line in llm_content.strip().split("\n"):
        line_content = line.strip()
        if not line_content:
            continue

        if line_content.startswith("OUTPUT:"):
            # Save previous output if exists and complete
            if current_output_name and "type" in current_data and "description" in current_data:
                outputs_list.append({current_output_name: current_data})

            # Start new output
            current_output_name = line_content[len("OUTPUT:") :].strip()
            # Ensure name has no spaces and is lowercase
            current_output_name = current_output_name.replace(" ", "_").lower()
            current_data = {}  # Reset data for the new output

        elif current_output_name and line_content.startswith("TYPE:"):
            current_data["type"] = line_content[len("TYPE:") :].strip()

        elif current_output_name and line_content.startswith("DESCRIPTION:"):
            current_data["description"] = line_content[len("DESCRIPTION:") :].strip()

    # Save the very last output if it exists and is complete
    if current_output_name and "type" in current_data and "description" in current_data:
        outputs_list.append({current_output_name: current_data})

    return outputs_list


def generate_outputs(
    profiles: list[dict[str, Any]],
    functionalities: list[str],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generates and adds structured output definitions to user profiles using an LLM.

    For each user profile, this function:
    1. Sanitizes the profile's goals by replacing variable placeholders.
    2. Constructs a prompt asking the LLM to define relevant output fields based
       on the profile, goals, and known chatbot functionalities.
    3. Invokes the LLM with the prompt.
    4. Parses the LLM's response into a structured list of output definitions.
    5. Adds the generated 'outputs' list to the corresponding profile dictionary.

    Args:
        profiles: A list of dictionaries, where each dictionary represents a user profile.
                  Expected to have a 'goals' key (list of strings).
        functionalities: A list of strings describing the known functionalities of the chatbot.
        llm: An instance of a LangChain BaseLanguageModel used for generation.
        supported_languages: An optional list of languages. If provided, the first
                             language is used to instruct the LLM.

    Returns:
        The original list of profiles, with each profile dictionary now potentially
        containing an added 'outputs' key holding the list of generated output definitions.
    """
    primary_language = ""
    language_instruction = ""
    if supported_languages:
        primary_language = supported_languages[0]
        language_instruction = f"Write the descriptions in {primary_language}."

    for profile in profiles:
        # Sanitize goals for LLM prompt
        sanitized_goals = []
        for goal in profile.get("goals", []):
            if isinstance(goal, str):
                sanitized_goal = re.sub(VARIABLE_PATTERN, "[specific details]", goal)
                sanitized_goals.append(sanitized_goal)

        # Generate prompt and invoke LLM
        outputs_prompt = get_outputs_prompt(
            profile=profile,
            sanitized_goals=sanitized_goals,
            functionalities=functionalities,
            language_instruction=language_instruction,
        )
        outputs_response = llm.invoke(outputs_prompt)

        # Parse the LLM response using the helper function
        parsed_outputs = _parse_llm_outputs(outputs_response.content)

        # Store outputs in the profile
        profile["outputs"] = parsed_outputs

        logger.verbose("    Generated %d outputs", len(parsed_outputs))

    return profiles
