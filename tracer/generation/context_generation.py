"""Module to generate the context for the profiles."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.context_generation_prompts import get_context_prompt
from tracer.utils.logging_utils import get_logger

logger = get_logger()

MAX_CONTEXT_PREVIEW_LENGTH = 70


def _get_language_instruction(supported_languages: list[str] | None) -> str:
    """Get language instruction based on supported languages."""
    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        return f"Write the context in {primary_language}."
    return ""


def _extract_llm_content(llm_response_obj: object) -> str:
    """Extract content from LLM response object."""
    if hasattr(llm_response_obj, "content"):
        return llm_response_obj.content.strip()
    return str(llm_response_obj).strip()


def _parse_context_entries(context_content: str) -> list[str]:
    """Parse context entries from LLM response content."""
    context_entries = []
    for line in context_content.split("\n"):
        line_content = line.strip()
        if line_content.startswith("- "):
            entry = line_content[2:].strip()
            if entry:
                context_entries.append(entry)
    return context_entries


def _build_final_context(existing_context: list[Any] | None, context_entries: list[str]) -> list[str]:
    """Build final context list preserving personality entries."""
    final_context_list = []

    if isinstance(existing_context, list):
        # Use list comprehension for better performance
        personality_items = [
            item for item in existing_context if isinstance(item, str) and item.startswith("personality:")
        ]
        final_context_list.extend(personality_items)

    final_context_list.extend(context_entries)
    return final_context_list


def _generate_context_for_profile(
    profile: dict[str, Any], llm: BaseLanguageModel, language_instruction: str
) -> list[str]:
    """Generate context for a single profile."""
    profile_name = profile.get("name", "Unnamed Profile")
    logger.debug("Generating context for profile: '%s'", profile_name)

    context_prompt_str = get_context_prompt(
        profile=profile,
        language_instruction=language_instruction,
    )

    llm_response_obj = llm.invoke(context_prompt_str)
    context_content = _extract_llm_content(llm_response_obj)
    context_entries = _parse_context_entries(context_content)

    if not context_entries:
        logger.warning("LLM did not generate valid context points for '%s'. Using a default.", profile_name)
        context_entries = ["The user has some general inquiries or tasks to perform."]

    return context_entries


def generate_context(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate additional context for the user simulator as multiple short entries."""
    language_instruction = _get_language_instruction(supported_languages)

    for profile in profiles:
        profile_name = profile.get("name", "Unnamed Profile")
        context_entries = _generate_context_for_profile(profile, llm, language_instruction)

        existing_context = profile.get("context", [])
        final_context_list = _build_final_context(existing_context, context_entries)
        profile["context"] = final_context_list

        logger.verbose(
            "    Generated/updated context for profile '%s'. Total context entries: %d. First entry preview: %s",
            profile_name,
            len(context_entries),
            (
                context_entries[0][:MAX_CONTEXT_PREVIEW_LENGTH]
                + ("..." if len(context_entries[0]) > MAX_CONTEXT_PREVIEW_LENGTH else "")
            )
            if context_entries
            else "N/A",
        )

    return profiles
