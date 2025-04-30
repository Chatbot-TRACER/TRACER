"""Generates user profiles based on chatbot analysis results."""

import re
from typing import Any, TypedDict

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.generation.context_generation import generate_context
from chatbot_explorer.generation.output_generation import generate_outputs
from chatbot_explorer.generation.variable_definition import generate_variable_definitions
from chatbot_explorer.prompts.profile_generation_prompts import (
    get_language_instruction_goals,
    get_language_instruction_grouping,
    get_profile_goals_prompt,
    get_profile_grouping_prompt,
)


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
    """Ensures that all single curly braces in the text are replaced with double curly braces.

    Args:
        text: The input string potentially containing single curly braces.

    Returns:
        The string with single curly braces converted to double curly braces.
    """
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
    num_functionalities = len(config["functionalities"])
    min_profiles = 3
    max_profiles = 10
    suggested_profiles = max(min_profiles, min(max_profiles, num_functionalities))
    chatbot_type_context = f"CHATBOT TYPE: {config['chatbot_type'].upper()}\n"

    grouping_prompt = get_profile_grouping_prompt(
        functionalities=config["functionalities"],
        conversation_context=conv_context,
        workflow_context=wf_context,
        chatbot_type_context=chatbot_type_context,
        language_instruction_grouping=lang_instr,
        suggested_profiles=suggested_profiles,
    )

    profiles_response = config["llm"].invoke(grouping_prompt)
    return _parse_profile_groupings(profiles_response.content)


def _generate_profile_goals(
    profile: dict[str, Any], config: ProfileGenerationConfig, conv_context: str, wf_context: str, lang_instr: str
) -> list[str]:
    """Generates and parses goals for a single profile using the LLM."""
    chatbot_type_context = f"CHATBOT TYPE: {config['chatbot_type'].upper()}\n"
    goals_prompt = get_profile_goals_prompt(
        profile=profile,
        chatbot_type_context=chatbot_type_context,
        workflow_context=wf_context,
        limitations=config["limitations"],
        conversation_context=conv_context,
        language_instruction_goals=lang_instr,
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
    return goals


def generate_profile_content(config: ProfileGenerationConfig) -> list[dict[str, Any]]:
    """Generates the complete content for user profiles based on chatbot analysis.

    Args:
        config: A ProfileGenerationConfig dictionary containing all necessary inputs
                like functionalities, limitations, LLM instance, history, etc.

    Returns:
        A list of dictionaries, where each dictionary represents a fully generated
        user profile including name, role, functionalities, goals, variables,
        context, and outputs.
    """
    # 1. Prepare context strings
    lang_instr_group, lang_instr_goal = _prepare_language_instructions(config["supported_languages"])
    conv_context = _prepare_conversation_context(config["conversation_history"])
    workflow_context = _prepare_workflow_context(config["workflow_structure"])

    # 2. Generate initial profile groupings (name, role, functionalities)
    profiles = _generate_profile_groupings(config, conv_context, workflow_context, lang_instr_group)

    # 3. Generate goals for each profile
    for profile in profiles:
        profile["goals"] = _generate_profile_goals(profile, config, conv_context, workflow_context, lang_instr_goal)

    # 4. Generate variable definitions based on goals
    profiles = generate_variable_definitions(profiles, config["llm"], config["supported_languages"])

    # 5. Generate context based on variables
    profiles = generate_context(profiles, config["llm"], config["supported_languages"])

    # 6. Generate output fields based on profiles and functionalities
    return generate_outputs(profiles, config["functionalities"], config["llm"], config["supported_languages"])
