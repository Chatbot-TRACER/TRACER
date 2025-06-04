"""Module to generate the context for the profiles."""

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.context_generation_prompts import get_context_prompt
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

MAX_CONTEXT_PREVIEW_LENGTH = 70

def generate_context(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate additional context for the user simulator as multiple short entries."""
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Write the context in {primary_language}."

    for profile in profiles:
        profile_name = profile.get("name", "Unnamed Profile")
        logger.debug(f"Generating context for profile: '{profile_name}'")

        context_prompt_str = get_context_prompt(
            profile=profile,
            language_instruction=language_instruction,
        )


        llm_response_obj = llm.invoke(context_prompt_str)
        context_content = ""
        if hasattr(llm_response_obj, "content"):
            context_content = llm_response_obj.content.strip()
        else:
            context_content = str(llm_response_obj).strip()


        context_entries = []
        for line in context_content.split("\n"):
            line_content = line.strip()
            if line_content.startswith("- "):
                entry = line_content[2:].strip()
                if entry:
                    context_entries.append(entry)

        if not context_entries:
            logger.warning(f"LLM did not generate valid context points for '{profile_name}'. Using a default.")
            context_entries = ["The user has some general inquiries or tasks to perform."]

        final_context_list = []
        existing_context = profile.get("context", [])

        if isinstance(existing_context, list):
            for item in existing_context:
                if isinstance(item, str) and item.startswith("personality:"):
                    final_context_list.append(item)

        final_context_list.extend(context_entries)

        profile["context"] = final_context_list

        logger.verbose(
            "    Generated/updated context for profile '%s'. Total context entries: %d. First entry preview: %s",
            profile_name,
            len(context_entries),
            (context_entries[0][:MAX_CONTEXT_PREVIEW_LENGTH] + ("..." if len(context_entries[0]) > MAX_CONTEXT_PREVIEW_LENGTH else "")) if context_entries else "N/A",
        )

    return profiles
