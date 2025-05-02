"""Node for generating conversation parameters (number, cost, style) for user profiles."""

import contextlib
from typing import Any, TypedDict

from langchain_core.language_models.base import BaseLanguageModel

from chatbot_explorer.prompts.conversation_params_prompts import (
    PromptLanguageSupport,
    PromptPreviousParams,
    PromptProfileContext,
    get_goal_style_prompt,
    get_interaction_style_prompt,
    get_max_cost_prompt,
    get_number_prompt,
)
from chatbot_explorer.schemas.graph_state_model import State
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()

# --- Helper Functions for extract_profile_variables ---


def _get_profile_variables(profile: dict[str, Any]) -> list[str]:
    """Extracts all defined variable names from a profile."""
    return [
        var_name
        for var_name, var_def in profile.items()
        if isinstance(var_def, dict) and "function" in var_def and "data" in var_def
    ]


def _calculate_combinations(profile: dict[str, Any], variables: list[str]) -> int:
    """Calculates the potential number of combinations based on variable definitions."""
    combinations = 1
    for var_name in variables:
        var_def = profile.get(var_name, {})
        if isinstance(var_def, dict) and "data" in var_def:
            data = var_def.get("data", [])
            if isinstance(data, list):
                combinations *= len(data) if data else 1
            elif isinstance(data, dict) and all(k in data for k in ["min", "max", "step"]) and data["step"] != 0:
                steps = (data["max"] - data["min"]) / data["step"] + 1
                combinations *= int(steps) if steps >= 1 else 1
    return combinations


def _check_nested_forwards(profile: dict[str, Any], variables: list[str]) -> tuple[bool, list[str], str]:
    """Checks for nested forward dependencies and calculates related info."""
    has_nested_forwards = profile.get("has_nested_forwards", False)
    forward_with_dependencies = []
    nested_forward_info = ""

    if "forward_dependencies" in profile:
        forward_dependencies = profile["forward_dependencies"]
        forward_with_dependencies = list(forward_dependencies.keys())

        if has_nested_forwards and "nested_forward_chains" in profile:
            nested_chains = profile["nested_forward_chains"]
            chain_descriptions = [f"Chain: {' → '.join(chain)}" for chain in nested_chains]

            if chain_descriptions:
                nested_forward_info = "\nNested dependency chains detected:\n" + "\n".join(chain_descriptions)
                combinations = _calculate_combinations(profile, variables)
                nested_forward_info += f"\nPotential combinations: approximately {combinations}"
    else:  # Fallback if structured dependencies aren't present
        for var_name, var_def in profile.items():
            if (
                isinstance(var_def, dict)
                and "function" in var_def
                and "forward" in var_def["function"]
                and "(" in var_def["function"]
                and ")" in var_def["function"]
            ):
                param = var_def["function"].split("(")[1].split(")")[0]
                if param and param != "rand" and not param.isdigit():
                    forward_with_dependencies.append(var_name)

    return has_nested_forwards, forward_with_dependencies, nested_forward_info


def _build_variables_info_string(
    variables: list[str],
    forward_with_dependencies: list[str],
    nested_forward_info: str,
    *,
    has_nested_forwards: bool,
) -> str:
    """Builds the descriptive string about variables for LLM prompts."""
    if not variables:
        return ""

    variables_info = f"\nThis profile has {len(variables)} variables: {', '.join(variables)}"
    if forward_with_dependencies:
        variables_info += (
            f"\n{len(forward_with_dependencies)} variables have dependencies: {', '.join(forward_with_dependencies)}"
        )
        if has_nested_forwards:
            variables_info += "\nThis creates COMBINATIONS that could be explored with 'all_combinations', 'sample(X)', or a fixed number."
            variables_info += f"\nIMPORTANT: This profile has NESTED FORWARD DEPENDENCIES.{nested_forward_info}"
    return variables_info


def extract_profile_variables(profile: dict[str, Any]) -> tuple[list[str], list[str], bool, str, str]:
    """Extracts variables, dependency info, and builds a descriptive string from a profile.

    Args:
        profile: The user profile dictionary.

    Returns:
        A tuple containing:
            - List of all variable names.
            - List of variables with forward dependencies.
            - Boolean indicating if nested forwards exist.
            - String with details about nested forward chains and combinations.
            - A combined descriptive string about variables for LLM prompts.
    """
    variables = _get_profile_variables(profile)
    has_nested_forwards, forward_with_dependencies, nested_forward_info = _check_nested_forwards(profile, variables)
    variables_info = _build_variables_info_string(
        variables, forward_with_dependencies, nested_forward_info, has_nested_forwards=has_nested_forwards
    )
    return variables, forward_with_dependencies, has_nested_forwards, nested_forward_info, variables_info


# --- Language Info Preparation ---


def prepare_language_info(supported_languages: list[str] | None) -> tuple[str, str, str]:
    """Prepares language-related strings for LLM prompts."""
    language_info = ""
    languages_example = ""
    supported_languages_text = ""

    if supported_languages:
        language_info = f"\nSUPPORTED LANGUAGES: {', '.join(supported_languages)}"
        supported_languages_text = f"({', '.join(supported_languages)})"
        languages_example = "\n".join([f"- {lang.lower()}" for lang in supported_languages])

    return language_info, languages_example, supported_languages_text


# --- Context for Parameter Requests ---


class ParamRequestContext(TypedDict):
    """Context dictionary for requesting conversation parameters."""

    llm: BaseLanguageModel
    profile: dict[str, Any]
    variables_info: str
    language_info: str
    supported_languages_text: str
    languages_example: str


# --- LLM Request Functions ---


def _parse_number_response(response_text: str, default_number: str | int) -> str | int:
    """Parses the LLM response for the 'number' parameter."""
    extracted_number = None
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "number":
            value = value.strip()
            if value == "all_combinations":
                extracted_number = "all_combinations"
            elif "sample" in value.lower() and "(" in value and ")" in value:
                with contextlib.suppress(ValueError):
                    sample_value = float(value.split("(")[1].split(")")[0])
                    min_sample_value = 0.1
                    max_sample_value = 1.0
                    if min_sample_value <= sample_value <= max_sample_value:
                        extracted_number = f"sample({sample_value})"
            elif value.isdigit():
                extracted_number = int(value)
            break  # Found number, no need to check further lines
    return extracted_number if extracted_number is not None else default_number


def request_number_from_llm(context: ParamRequestContext, variables: list[str]) -> str | int:
    """Prompts LLM to determine the NUMBER parameter."""
    default_number = "sample(0.2)" if context["has_nested_forwards"] else (5 if variables else 2)
    prompt = get_number_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        has_nested_forwards=context["has_nested_forwards"],
    )
    response = context["llm"].invoke(prompt)
    return _parse_number_response(response.content.strip(), default_number)


def request_max_cost_from_llm(context: ParamRequestContext, number_value: str | int) -> float:
    """Prompts LLM to determine the MAX_COST parameter."""
    prompt = get_max_cost_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        number_value=number_value,
    )
    response = context["llm"].invoke(prompt)
    response_text = response.content.strip()
    default_cost = 1.0 if not context["has_nested_forwards"] else 2.0
    max_cost = default_cost
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "max_cost":
            with contextlib.suppress(ValueError):
                max_cost = float(value.strip())
            break
    return max_cost


def _parse_goal_style_response(response_text: str) -> dict[str, Any]:
    """Parses the LLM response for the 'goal_style' parameter."""
    goal_style = {"steps": 11}  # Default
    selected_style = "steps"
    goal_style_value = 11

    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "goal_style" and value in ["steps", "all_answered", "random_steps"]:
            selected_style = value
        elif key == "goal_style_value":
            with contextlib.suppress(ValueError):
                goal_style_value = int(value)

        # Update goal_style based on potentially updated selected_style and goal_style_value
        if selected_style == "steps":
            goal_style = {"steps": goal_style_value}
        elif selected_style == "all_answered":
            goal_style = {"all_answered": {"export": False, "limit": goal_style_value}}
        elif selected_style == "random_steps":
            goal_style = {"random_steps": goal_style_value}

    return goal_style


def request_goal_style_from_llm(
    context: ParamRequestContext, number_value: str | int, max_cost: float
) -> dict[str, Any]:
    """Prompts LLM to determine the GOAL_STYLE parameter."""
    prompt = get_goal_style_prompt(
        profile=context["profile"],
        variables_info=context["variables_info"],
        language_info=context["language_info"],
        number_value=number_value,
        max_cost=max_cost,
    )
    response = context["llm"].invoke(prompt)
    return _parse_goal_style_response(response.content.strip())


def request_interaction_style_from_llm(
    context: ParamRequestContext, number_value: str | int, max_cost: float, goal_style: dict[str, Any]
) -> list[str]:
    """Prompts LLM to determine the INTERACTION_STYLE parameter."""
    profile_context: PromptProfileContext = {
        "profile": context["profile"],
        "variables_info": context["variables_info"],
        "language_info": context["language_info"],
    }
    prev_params: PromptPreviousParams = {
        "number_value": number_value,
        "max_cost": max_cost,
        "goal_style": goal_style,
    }
    lang_support: PromptLanguageSupport = {
        "supported_languages_text": context["supported_languages_text"],
        "languages_example": context["languages_example"],
    }

    # Call the prompt function with the structured arguments
    prompt = get_interaction_style_prompt(
        profile_context=profile_context,
        prev_params=prev_params,
        lang_support=lang_support,
    )
    response = context["llm"].invoke(prompt)
    response_text = response.content.strip()
    interaction_styles = []
    for line in response_text.split("\n"):
        line_content = line.strip()
        if not line_content or ":" not in line_content:
            continue
        key, value = line_content.split(":", 1)
        if key.strip().lower() == "interaction_style":
            value = value.strip()
            if "[" in value and "]" in value:
                styles_part = value.replace("[", "").replace("]", "")
                styles = [s.strip().strip("\"'") for s in styles_part.split(",")]
                interaction_styles = [s for s in styles if s]
            else:
                style = value.strip().strip("\"'")
                if style:
                    interaction_styles.append(style)
            break
    return interaction_styles


# --- Main Generation Function ---


def generate_conversation_parameters(
    profiles: list[dict[str, Any]],
    llm: BaseLanguageModel,
    supported_languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Generates conversation parameters for each profile using sequential LLM prompting.

    Args:
        profiles: List of user profile dictionaries.
        llm: The language model instance.
        supported_languages: Optional list of supported languages.

    Returns:
        The list of profiles with an added 'conversation' key containing the generated parameters.
    """
    language_info, languages_example, supported_languages_text = prepare_language_info(supported_languages)

    # Process each profile
    total_profiles = len(profiles)
    for i, profile in enumerate(profiles, 1):
        profile_name = profile.get("name", f"Profile {i}")
        logger.verbose("Processing profile %d/%d: '%s'", i, total_profiles, profile_name)

        # Extract variables information
        variables, forward_vars, has_nested_forwards, _, variables_info = extract_profile_variables(profile)

        if variables:
            var_summary = ", ".join(variables)
            logger.debug("Profile has %d variables: %s", len(variables), var_summary)

            if forward_vars:
                logger.debug(
                    "With %d dependent variables: %s",
                    len(forward_vars),
                    ", ".join(forward_vars),
                )
        else:
            logger.debug("Profile has no variables")

        # Create context for this profile's requests
        request_context: ParamRequestContext = {
            "llm": llm,
            "profile": profile,
            "variables_info": variables_info,
            "language_info": language_info,
            "has_nested_forwards": has_nested_forwards,
            "supported_languages_text": supported_languages_text,
            "languages_example": languages_example,
        }

        # Sequential prompting with minimal but sufficient logging
        logger.debug("Determining parameters for profile '%s'", profile_name)

        # Fetch parameters sequentially with minimal logging
        number_value = request_number_from_llm(request_context, variables)
        max_cost = request_max_cost_from_llm(request_context, number_value)
        goal_style = request_goal_style_from_llm(request_context, number_value, max_cost)
        interaction_styles = request_interaction_style_from_llm(request_context, number_value, max_cost, goal_style)

        # Log a concise summary of the key parameters
        goal_style_type = next(iter(goal_style.keys())) if goal_style else "unknown"
        interaction_style_summary = ""
        if interaction_styles:
            interaction_style_summary = f", styles: {', '.join(interaction_styles[:2])}" + (
                f" +{len(interaction_styles) - 2} more" if len(interaction_styles) > 2 else ""
            )

        logger.debug(
            "Parameters: number=%s, cost=%.2f, goal=%s%s",
            number_value,
            max_cost,
            goal_style_type,
            interaction_style_summary,
        )

        # Build and assign parameters
        conversation_params = {"number": number_value, "max_cost": max_cost, "goal_style": goal_style}
        if interaction_styles:
            conversation_params["interaction_style"] = interaction_styles
        profile["conversation"] = conversation_params

    return profiles


# --- LangGraph Node ---


def conversation_params_node(state: State, llm: BaseLanguageModel) -> dict[str, Any]:
    """Node that generates specific parameters needed for conversation goals."""
    conversation_goals = state.get("conversation_goals")
    if not conversation_goals:
        logger.info("Skipping conversation parameters: No goals generated.")
        return {"conversation_goals": []}

    logger.info("\nStep 3: Conversation parameters generation")
    logger.info("------------------------------------------")

    # Flatten functionalities (currently unused by generate_conversation_parameters but kept for context)
    structured_root_dicts = state.get("discovered_functionalities", [])
    flat_func_info = []
    nodes_to_process = list(structured_root_dicts)
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        info = {k: v for k, v in node.items() if k != "children"}
        flat_func_info.append(info)
        if node.get("children"):
            nodes_to_process.extend(node["children"])

    try:
        # Initial progress message
        total_profiles = len(conversation_goals)
        logger.verbose("\nGenerating parameters for %d profiles:", total_profiles)

        # Generate parameters
        profiles_with_params = generate_conversation_parameters(
            conversation_goals,
            llm,
            supported_languages=state.get("supported_languages"),
        )

        # Simple completion message (no need to list all profiles again)
        logger.info("Successfully generated conversation parameters for all %d profiles", len(profiles_with_params))

    except Exception:
        logger.exception("Error during parameter generation")
        return {"conversation_goals": conversation_goals}
    else:
        return {"conversation_goals": profiles_with_params}
