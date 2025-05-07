"""Generates user profiles based on chatbot analysis results."""

import re
from typing import Any, TypedDict

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.generation.context_generation import generate_context
from chatbot_explorer.generation.output_generation import generate_outputs
from chatbot_explorer.generation.variable_definition import generate_variable_definitions
from chatbot_explorer.prompts.profile_generation_prompts import (
    ProfileGoalContext,
    ProfileGroupingContext,
    get_language_instruction_goals,
    get_language_instruction_grouping,
    get_profile_goals_prompt,
    get_profile_grouping_prompt,
)
from chatbot_explorer.utils.logging_utils import get_logger

MAX_DISPLAYED_FUNCS = 3
MAX_DEFINITION_LENGTH = 70

logger = get_logger()


class ProfileGenerationConfig(TypedDict):
    """Configuration for generating user profiles.

    Arguments:
        functionalities: A list of strings describing the known functionalities of the chatbot.
        limitations: A list of strings describing the known limitations of the chatbot.
        llm: An instance of the BaseLanguageModel to interact with.
        workflow_structure: A list of dictionaries defining the workflow structure or None if not applicable.
        conversation_history: A list of conversation history or None if not available.
        supported_languages: A list of supported languages or None if not specified.
        chatbot_type: A string indicating the type of chatbot.
    """

    functionalities: list[str]
    limitations: list[str]
    llm: BaseLanguageModel
    workflow_structure: list[dict[str, Any]] | None
    conversation_history: list[list[dict[str, str]]] | None
    supported_languages: list[str] | None
    chatbot_type: str


def ensure_double_curly(text: str) -> str:
    """Ensures that all single curly braces in the text are replaced with double curly braces."""
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def _prepare_language_instructions(
    supported_languages: list[str] | None,
) -> tuple[str, str]:
    """Prepares language instruction strings based on supported languages."""
    primary_language = ""
    language_instruction_grouping = ""
    language_instruction_goals = ""
    if supported_languages:
        primary_language = supported_languages[0]
        language_instruction_grouping = get_language_instruction_grouping(primary_language)
        language_instruction_goals = get_language_instruction_goals(primary_language)
    return language_instruction_grouping, language_instruction_goals


def _prepare_conversation_context(
    conversation_history: list[list[dict[str, str]]] | None,
) -> str:
    """Formats conversation history into a string context for the LLM."""
    if not conversation_history:
        return ""
    context = "Here are some example conversations with the chatbot:\n\n"
    for i, session in enumerate(conversation_history, 1):
        context += f"--- SESSION {i} ---\n"
        for turn in session:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "assistant":  # Explorer AI
                context += f"Human: {content}\n"
            elif role == "user":  # Chatbot's response
                context += f"Chatbot: {content}\n"
        context += "\n"
    return context


def _prepare_workflow_context(workflow_structure: list[dict[str, Any]] | None) -> str:
    """Formats workflow structure into a string context for the LLM."""
    if not workflow_structure:
        return ""
    context = "WORKFLOW INFORMATION (how functionalities connect):\n"
    for node in workflow_structure:
        if isinstance(node, dict):
            node_name = node.get("name")
            node_children = node.get("children", [])
            if node_name and node_children:
                child_names = [
                    child.get("name") for child in node_children if isinstance(child, dict) and child.get("name")
                ]
                if child_names:
                    context += f"- {node_name} can lead to: {', '.join(child_names)}\n"
            elif node_name:
                context += f"- {node_name} (standalone functionality)\n"
    return context


def _parse_profile_groupings(llm_content: str) -> list[dict[str, Any]]:
    """Parses the LLM response containing profile groupings."""
    profiles = []
    profile_sections = llm_content.split("## PROFILE:")
    if not profile_sections[0].strip():  # Handle potential empty first split
        profile_sections = profile_sections[1:]

    for section in profile_sections:
        lines = section.strip().split("\n")
        if not lines:
            continue
        profile_name = lines[0].strip()
        role = ""
        functionalities_list = []
        role_started = False
        func_started = False
        for line in lines[1:]:
            line_strip = line.strip()
            if line_strip.startswith("ROLE:"):
                role_started = True
                role = line_strip[len("ROLE:") :].strip()
                func_started = False
            elif line_strip.startswith("FUNCTIONALITIES:"):
                role_started = False
                func_started = True
            elif func_started and line_strip.startswith("- "):
                functionalities_list.append(line_strip[2:].strip())
            elif role_started:  # Continue multi-line role description
                role += " " + line_strip
        if profile_name:  # Only add if a profile name was found
            profiles.append(
                {
                    "name": profile_name,
                    "role": role.strip(),
                    "functionalities": functionalities_list,
                }
            )
    return profiles


def _generate_profile_groupings(
    config: ProfileGenerationConfig, conv_context: str, wf_context: str, lang_instr: str
) -> list[dict[str, Any]]:
    """Given the chatbot's functionalities, generates user profiles using the LLM."""
    logger.verbose("Generating initial profile groupings")

    num_functionalities = len(config["functionalities"])
    min_profiles = 3
    max_profiles = 10
    suggested_profiles = max(min_profiles, min(max_profiles, num_functionalities))
    chatbot_type_context = f"CHATBOT TYPE: {config['chatbot_type'].upper()}\n"

    grouping_context: ProfileGroupingContext = {
        "conversation_context": conv_context,
        "workflow_context": wf_context,
        "chatbot_type_context": chatbot_type_context,
        "language_instruction_grouping": lang_instr,
    }

    grouping_prompt = get_profile_grouping_prompt(
        functionalities=config["functionalities"],
        context=grouping_context,
        suggested_profiles=suggested_profiles,
    )

    profiles_response = config["llm"].invoke(grouping_prompt)
    profiles = _parse_profile_groupings(profiles_response.content)

    logger.verbose("Created %d initial profile groupings", len(profiles))
    return profiles


def _generate_profile_goals(
    profile: dict[str, Any], config: ProfileGenerationConfig, conv_context: str, wf_context: str, lang_instr: str
) -> list[str]:
    """Generates and parses goals for a single profile using the LLM."""
    profile_name = profile.get("name", "Unnamed profile")

    chatbot_type_context = f"CHATBOT TYPE: {config['chatbot_type'].upper()}\n"

    goal_context: ProfileGoalContext = {
        "chatbot_type_context": chatbot_type_context,
        "workflow_context": wf_context,
        "limitations": config["limitations"],
        "conversation_context": conv_context,
        "language_instruction_goals": lang_instr,
    }

    goals_prompt = get_profile_goals_prompt(
        profile=profile,
        context=goal_context,
    )

    goals_response = config["llm"].invoke(goals_prompt)
    goals_content = goals_response.content
    goals = []
    if "GOALS:" in goals_content:
        goals_section = goals_content.split("GOALS:")[1].strip()
        for line in goals_section.split("\n"):
            if line.strip().startswith("- "):
                goal = line.strip()[2:].strip().strip("\"'")
                if goal:
                    goal = ensure_double_curly(goal)
                    goals.append(goal)

    logger.verbose("    Generated %d goals for '%s'", len(goals), profile_name)

    # Log the goals in debug
    if logger.isEnabledFor(10):  # DEBUG level
        for i, goal in enumerate(goals[:3], 1):
            logger.debug("   - Goal %d: %s", i, goal)

    return goals


def generate_profile_content(config: ProfileGenerationConfig) -> list[dict[str, Any]]:
    """Generates the complete content for user profiles based on chatbot analysis."""
    logger.debug("Starting profile content generation")

    # 1. Prepare context strings
    lang_instr_group, lang_instr_goal = _prepare_language_instructions(config["supported_languages"])
    conv_context = _prepare_conversation_context(config["conversation_history"])
    workflow_context = _prepare_workflow_context(config["workflow_structure"])

    # 2. Generate initial profile groupings (name, role, functionalities)
    profiles = _generate_profile_groupings(config, conv_context, workflow_context, lang_instr_group)

    logger.info("Generating profile parameters for %d profiles:\n", len(profiles))

    # Process each profile with logging
    for i, profile in enumerate(profiles):
        profile_name = profile.get("name", f"Profile {i + 1}")
        logger.verbose("\n • Generating profile: %s", profile_name)

        # Log profile details in verbose mode
        if logger.isEnabledFor(15):  # VERBOSE level
            role = profile.get("role", "No role defined")
            funcs = profile.get("functionalities", [])
            func_count = len(funcs)
            logger.verbose("    Role: %s", role)
            logger.verbose("    Assigned functionalities: %d", func_count)

            if funcs and logger.isEnabledFor(10):  # DEBUG level
                for j, func in enumerate(funcs[:3], 1):
                    logger.debug("   - Functionality %d: %s", j, func)
                if len(funcs) > MAX_DISPLAYED_FUNCS:
                    logger.debug("   - ... plus %d more functionalities", len(funcs) - MAX_DISPLAYED_FUNCS)

        # 3. Generate goals for each profile
        profile["goals"] = _generate_profile_goals(profile, config, conv_context, workflow_context, lang_instr_goal)

        # 4. Generate variable definitions based on goals
        temp_profiles = generate_variable_definitions(
            [profile],
            config["llm"],
            config["supported_languages"],
            config["workflow_structure"],
        )
        if temp_profiles:
            profile.update(temp_profiles[0])

        # 5. Generate context based on variables
        temp_profiles = generate_context([profile], config["llm"], config["supported_languages"])
        if temp_profiles:
            profile.update(temp_profiles[0])

        # 6. Generate output fields
        temp_profiles = generate_outputs(
            [profile], config["functionalities"], config["llm"], config["supported_languages"]
        )
        if temp_profiles:
            profile.update(temp_profiles[0])

        # Show completion message
        logger.info(" ✅ Completed content for profile: %s", profile.get("name", "Unnamed profile"))
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("Final parameter values for profile '%s':", profile.get("name"))
            for g in profile.get("goals", []):
                if isinstance(g, dict):
                    for var_name, var_def in g.items():
                        if isinstance(var_def, dict) and "data" in var_def:
                            data = var_def.get("data", [])
                            if isinstance(data, list):
                                logger.debug(" → %s: %s", var_name, data)

    logger.debug("Completed generation of %d profiles", len(profiles))
    return profiles
