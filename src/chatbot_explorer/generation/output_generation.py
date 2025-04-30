import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import VARIABLE_PATTERN
from chatbot_explorer.prompts.output_generation_prompts import get_outputs_prompt


def generate_outputs(
    profiles: list[dict[str, Any]],
    functionalities: list[str],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate output fields to extract from chatbot responses."""
    # Work in the detected primary language
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Write the descriptions in {primary_language}."

    for profile in profiles:
        # Replace variables in goals with general terms for better LLM understanding
        sanitized_goals = []
        for goal in profile.get("goals", []):
            # Replace {{variable}} with general terms
            if isinstance(goal, str):
                sanitized_goal = re.sub(VARIABLE_PATTERN, "[specific details]", goal)
                sanitized_goals.append(sanitized_goal)

        outputs_prompt = get_outputs_prompt(
            profile=profile,
            sanitized_goals=sanitized_goals,
            functionalities=functionalities,
            language_instruction=language_instruction,
        )

        outputs_response = llm.invoke(outputs_prompt)

        # Parse the outputs
        outputs_list = []
        current_output = None
        current_data = {}

        for line in outputs_response.content.strip().split("\n"):
            line_content = line.strip()
            if not line_content:
                continue

            if line_content.startswith("OUTPUT:"):
                # Save previous output if exists
                if current_output and current_data:
                    outputs_list.append({current_output: current_data})

                # Start new output
                current_output = line_content[len("OUTPUT:") :].strip()
                # Ensure name has no spaces and is lowercase
                current_output = current_output.replace(" ", "_").lower()
                current_data = {}

            elif line_content.startswith("TYPE:"):
                current_data["type"] = line_content[len("TYPE:") :].strip()

            elif line_content.startswith("DESCRIPTION:"):
                current_data["description"] = line_content[len("DESCRIPTION:") :].strip()

        # Save last output
        if current_output and current_data:
            outputs_list.append({current_output: current_data})

        # Store outputs in the profile
        profile["outputs"] = outputs_list

    return profiles
