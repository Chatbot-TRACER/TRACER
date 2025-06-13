"""Prompts for generating user simulator context based on profile information."""

from typing import Any

from tracer.constants import LIST_TRUNCATION_THRESHOLD


def get_context_prompt(
    profile: dict[str, Any],
    language_instruction: str,
) -> str:
    """Generate the prompt for creating context points for a user profile."""
    profile_name = profile.get("name", "Unnamed Profile")
    profile_role = profile.get("role", "Unknown Role")

    goals_for_prompt_display = []
    variables_for_prompt_display = []
    for item in profile.get("goals", []):
        if isinstance(item, str):
            goals_for_prompt_display.append(f"- {item}")
        elif isinstance(item, dict):  # Variable definition
            for var_name, var_def in item.items():
                if isinstance(var_def, dict) and "data" in var_def:
                    data_preview = str(var_def.get("data"))
                    if (
                        isinstance(var_def.get("data"), list)
                        and len(var_def.get("data", [])) > LIST_TRUNCATION_THRESHOLD
                    ):
                        data_preview = f"{str(var_def.get('data', [])[:3])[:-1]}, ...]"  # Truncate list preview
                    elif isinstance(var_def.get("data"), dict):  # For numeric ranges
                        data_preview = f"min: {var_def['data'].get('min')}, max: {var_def['data'].get('max')}, step: {var_def['data'].get('step')}"
                    variables_for_prompt_display.append(
                        f"  - Note: Goal(s) may use '{{{{{var_name}}}}}' which iterates through values like: {data_preview}"
                    )

    goals_str = "\n".join(goals_for_prompt_display)
    vars_str = (
        "\n".join(variables_for_prompt_display)
        if variables_for_prompt_display
        else "  (No variables with predefined option lists found in goals for this profile)"
    )

    return f"""
    You are creating a background context scenario for a user simulator that will interact with a chatbot.
    This scenario, described in 2-3 lines, should provide a plausible motivation or situation for the user, aligning with their role and overall goals, but WITHOUT pre-determining specific choices that are meant to be covered by `{{variables}}` in the goals.

    CONVERSATION SCENARIO (PROFILE NAME): {profile_name}
    USER ROLE: {profile_role}

    USER GOALS (the simulator will try to achieve these; note any `{{variables}}`):
    {goals_str}

    VARIABLE DETAILS (these show how `{{variables}}` in goals will be iterated):
    {vars_str}

    {language_instruction}

    **GUIDELINES FOR CREATING A BACKGROUND CONTEXT SCENARIO (2-3 lines total):**
    1.  **Coherent Scenario:** The 2-3 lines together should describe a single, coherent situation, preference, constraint, or piece of background knowledge for the user. Each line should build upon the previous one to form this unified scenario.
    2.  **Support the Goals, Don't Pre-empt Variables:** The context should make the user's goals plausible but remain general. For example, if a goal is "Order an `{{item_selection}}`", a context point could be "You are looking for a suitable gift for a colleague," NOT "You want to order a blue widget." The `{{item_selection}}` variable will handle specific item choices.
    3.  **Compatibility with Variables:** Ensure context points are general enough to be true regardless of which specific value a variable (like `{{item_selection}}` or `{{item_size_preference}}`) takes during a test run.
        *   BAD Example (conflicts with variables): "You always order the 'deluxe' service package." (What if `{{service_package}}` iterates through 'basic', 'standard', 'deluxe'?)
        *   GOOD Example (compatible): "You are evaluating different service packages for your company's needs."
        *   GOOD Example: "You need the chosen service to be activated by next Monday."
    4.  **Concise and Flowing:** Each line should be 1-2 sentences. The lines should flow naturally as a single narrative, concisely describing the overall scenario.
    5.  **Realistic User Situations:** Focus on real-world scenarios.
    6.  **AVOID Specific Values Meant for Variables:** Do NOT put specific item names, sizes, colors, quantities, etc., in the context if those aspects are covered by `{{variables}}` in the goals. Use general statements.

    Example of a GOOD GENERALIZED context scenario (2-3 lines) for a service inquiry:
    - You've been tasked with finding a reliable service provider for a new project and are working with a specific budget.
    - A quick turnaround time is also a critical factor in your decision-making process for this service.

    FORMAT YOUR RESPONSE AS A SINGLE SCENARIO ACROSS 2-3 LINES:
    - First line setting up the scenario.
    - Second line adding detail or a related aspect to the same scenario.
    - Third line (optional) further elaborating or concluding the scenario.

    IMPORTANT: Each line MUST start with "- ". Do not add any other introductory or concluding text.
    """
