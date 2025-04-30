import re

from chatbot_explorer.constants import VARIABLE_PATTERN

# --- Context Generation Logic ---


def generate_context(profiles, functionalities, llm, supported_languages=None):
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

        context_prompt = f"""
        Create 2-3 SHORT context points for a user simulator interacting with a chatbot.
        Each point should be a separate piece of background information or context that helps the simulator.

        CONVERSATION SCENARIO: {profile["name"]}
        CURRENT USER ROLE: {profile["role"]}
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

        context_response = llm.invoke(context_prompt)
        context_content = context_response.content.strip()

        # Process context into separate entries
        context_entries = []
        for line in context_content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                entry = line[2:].strip()
                if entry:
                    context_entries.append(entry)

        profile["context"] = context_entries

    return profiles
