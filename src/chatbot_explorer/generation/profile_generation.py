import re
from typing import Any

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


def ensure_double_curly(text: str) -> str:
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def generate_profile_content(
    functionalities: list[str],
    limitations: list[str],
    llm: BaseLanguageModel,
    workflow_structure: list[dict[str, Any]] | None = None,
    conversation_history: list[list[dict[str, str]]] | None = None,
    supported_languages: list[str] | None = None,
    chatbot_type: str = "unknown",
) -> list[dict[str, Any]]:
    """Generates the complete content for user profiles."""
    # Work in the given language with stronger instructions
    primary_language = ""
    language_instruction_grouping = ""
    language_instruction_goals = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        # More specific instruction with examples that will help the model follow format
        language_instruction_grouping = get_language_instruction_grouping(primary_language)
        language_instruction_goals = get_language_instruction_goals(primary_language)

    # Prepare a condensed version of conversation history if available
    conversation_context = ""
    if conversation_history:
        conversation_context = "Here are some example conversations with the chatbot:\n\n"
        for i, session in enumerate(conversation_history, 1):
            conversation_context += f"--- SESSION {i} ---\n"
            for turn in session:
                if turn["role"] == "assistant":  # Explorer
                    conversation_context += f"Human: {turn['content']}\n"
                elif turn["role"] == "user":  # Chatbot's response
                    conversation_context += f"Chatbot: {turn['content']}\n"
            conversation_context += "\n"

    # Prepare workflow information if available
    workflow_context = ""
    if workflow_structure:
        workflow_context = "WORKFLOW INFORMATION (how functionalities connect):\n"
        # Process the workflow structure to extract relationships
        for node in workflow_structure:
            if isinstance(node, dict):
                node_name = node.get("name", "")
                node_children = node.get("children", [])
                if node_name and node_children:
                    child_names = [
                        child.get("name", "") for child in node_children if isinstance(child, dict) and "name" in child
                    ]
                    if child_names:
                        workflow_context += f"- {node_name} can lead to: {', '.join(child_names)}\n"
                elif node_name:
                    workflow_context += f"- {node_name} (standalone functionality)\n"

    # Include chatbot type information
    chatbot_type_context = f"CHATBOT TYPE: {chatbot_type.upper()}\n"

    # Ask the LLM to identify distinct conversation scenarios
    # Calculate an appropriate number of profiles based on functionality count
    num_functionalities = len(functionalities)
    min_profiles = 3
    max_profiles = 10
    suggested_profiles = max(min_profiles, min(max_profiles, num_functionalities))

    grouping_prompt = get_profile_grouping_prompt(
        functionalities=functionalities,
        conversation_context=conversation_context,
        workflow_context=workflow_context,
        chatbot_type_context=chatbot_type_context,
        language_instruction_grouping=language_instruction_grouping,
        suggested_profiles=suggested_profiles,
    )

    # Get scenario groupings from the LLM
    profiles_response = llm.invoke(grouping_prompt)
    profiles_content = profiles_response.content

    # Parse the profiles
    profile_sections = profiles_content.split("## PROFILE:")
    if not profile_sections[0].strip():
        profile_sections = profile_sections[1:]

    profiles = []
    for section in profile_sections:
        lines = section.strip().split("\n")
        profile_name = lines[0].strip()
        role = ""
        functionalities_list = []
        role_started = False
        func_started = False
        for line in lines[1:]:
            if line.startswith("ROLE:"):
                role_started = True
                role = line[len("ROLE:") :].strip()
                func_started = False
            elif line.startswith("FUNCTIONALITIES:"):
                role_started = False
                func_started = True
            elif func_started and line.strip().startswith("- "):
                functionalities_list.append(line.strip()[2:].strip())
            elif role_started:
                role += " " + line.strip()
        profiles.append(
            {
                "name": profile_name,
                "role": role.strip(),
                "functionalities": functionalities_list,
            },
        )

    # For each profile, generate user-centric goals
    for profile in profiles:
        goals_prompt = get_profile_goals_prompt(
            profile=profile,
            chatbot_type_context=chatbot_type_context,
            workflow_context=workflow_context,
            limitations=limitations,
            conversation_context=conversation_context,
            language_instruction_goals=language_instruction_goals,
        )

        goals_response = llm.invoke(goals_prompt)
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
        profile["goals"] = goals

    # Generate values for the variables
    profiles = generate_variable_definitions(profiles, llm, supported_languages)

    # Generate context
    profiles = generate_context(profiles, llm, supported_languages)

    # Generate output fields
    return generate_outputs(profiles, functionalities, llm, supported_languages)
