"""Prompts for generating user simulator context based on profile information."""

from typing import Any


def get_context_prompt(
    profile: dict[str, Any],
    sanitized_goals: list[str],
    language_instruction: str,
) -> str:
    """Generate the prompt for creating context points for a user profile."""
    return f"""
    Create 2-3 SHORT context points for a user simulator interacting with a chatbot.
    Each point should be a separate piece of background information or context that helps the simulator.

    CONVERSATION SCENARIO: {profile.get("name", "Unnamed Profile")}
    CURRENT USER ROLE: {profile.get("role", "Unknown Role")}
    USER GOALS (generalized):
    {", ".join(sanitized_goals)}

    {language_instruction}

    GUIDELINES:
    1. Write 2-3 SEPARATE short context points, each 1-2 sentences only
    2. Each point should focus on ONE aspect (background info, knowledge, motivation)
    3. NEVER include variables like {{date}} or {{amount}} - use specific examples instead
    4. Keep each point brief and focused
    5. Make each point distinctly different from the others

    Examples of GOOD context points:
    - "You tried calling the office yesterday but no one answered."
    - "You need to finish this task before your meeting at 3pm."
    - "Your colleague mentioned this service was very reliable."

    FORMAT YOUR RESPONSE WITH ONE CONTEXT POINT PER LINE:
    - First context point
    - Second context point
    - Third context point (optional)

    IMPORTANT: Each line should start with "- " and be a standalone piece of context.
    """
