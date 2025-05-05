"""Module to generate the context for the profiles."""

import re
from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.constants import VARIABLE_PATTERN
from chatbot_explorer.prompts.context_generation_prompts import get_context_prompt
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_LENGTH = 50


def generate_context(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate additional context for the user simulator as multiple short entries."""
    # Work in the detected primary language
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Write the context in {primary_language}."

    for profile in profiles:
        # Replace variables in goals with general terms for context generation
        sanitized_goals = []
        for goal in profile.get("goals", []):
            # Replace {{variable}} with general terms
            if isinstance(goal, str):
                sanitized_goal = re.sub(VARIABLE_PATTERN, "[specific details]", goal)
                sanitized_goals.append(sanitized_goal)

        context_prompt = get_context_prompt(
            profile=profile,
            sanitized_goals=sanitized_goals,
            language_instruction=language_instruction,
        )

        context_response = llm.invoke(context_prompt)
        context_content = context_response.content.strip()

        # Process context into separate entries
        context_entries = []
        for line in context_content.split("\n"):
            line_content = line.strip()
            if line_content.startswith("- "):
                entry = line_content[2:].strip()
                if entry:
                    context_entries.append(entry)

        if len(context_entries) == 0:
            # Fallback if no context entries generated
            context_entries = ["No specific context available."]
        else:
            logger.verbose(
                "    Generated %d context entries: %s",
                len(context_entries),
                context_entries[0][:MAX_LENGTH] + ("..." if len(context_entries[0]) > MAX_LENGTH else ""),
            )

        profile["context"] = context_entries

    return profiles
