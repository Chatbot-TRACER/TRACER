"""Prompts for generating conversation parameters based on user profile information."""

from typing import Any, TypedDict

# --- Data Structures for Prompt Arguments ---


class PromptProfileContext(TypedDict):
    """Context related to the user profile for prompts.

    Args:
        profile: The user profile dictionary.
        variables_info: Descriptive string about profile variables.
        language_info: Descriptive string about supported languages.
    """

    profile: dict[str, Any]
    variables_info: str
    language_info: str


class PromptPreviousParams(TypedDict):
    """Previously determined parameters for prompts.

    Args:
        number_value: The determined 'number' parameter (int or str).
        max_cost: The determined 'max_cost' parameter (float).
        goal_style: The determined 'goal_style' dictionary.
    """

    number_value: int | str
    max_cost: float
    goal_style: dict[str, Any]


class PromptLanguageSupport(TypedDict):
    """Language support information for prompts.

    Args:
        supported_languages_text: String listing supported languages (e.g., "English, Spanish").
        languages_example: Formatted string showing language examples for the prompt.
    """

    supported_languages_text: str
    languages_example: str


# --- Prompt Generation Functions ---


def get_number_prompt(
    profile: dict[str, Any],
    variables_info: str,
    language_info: str,
    *,
    has_nested_forwards: bool,
) -> str:
    """Generate the prompt to determine the NUMBER parameter.

    Args:
        profile: The user profile dictionary.
        variables_info: Descriptive string about profile variables.
        language_info: Descriptive string about supported languages.
        has_nested_forwards: Flag indicating if the profile has nested forward dependencies.

    Returns:
        The formatted prompt string for the LLM.
    """
    if has_nested_forwards:
        number_section = """
        NUMBER:
        Choose ONE option:
        - Enter a fixed number (e.g., 2-5) for a specific number of conversations.
        - "all_combinations" to try all possible combinations (ONLY if there are fewer than 5 total combinations).
        - "sample(X)" where X is a decimal between 0.1 and 1.0 to test a fraction of the combinations.

        RECOMMENDATION:
        - For nested forwards, if total combinations are low (<5), you may choose "all_combinations".
        - Otherwise, consider using "sample(0.2)" or "sample(0.5)" based on your testing needs, or simply specify a fixed number.
        """
    else:
        number_section = """
        NUMBER:
        Choose a specific number between 2-5 conversations to generate.

        RECOMMENDATION:
        - 2-3 for simple, straightforward conversations.
        - 4-5 for more complex scenarios with multiple user goals.
        """

    return f"""
    Determine the NUMBER parameter for this conversation scenario.

    CONVERSATION SCENARIO: {profile.get("name", "Unnamed")}
    USER ROLE: {profile.get("role", "Unknown")}
    {variables_info}{language_info}

    {number_section}

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    NUMBER: specific_number or all_combinations or sample(0.X)

    Examples:
    NUMBER: 3
    NUMBER: all_combinations
    NUMBER: sample(0.5)
    """


def get_max_cost_prompt(
    profile: dict[str, Any],
    variables_info: str,
    language_info: str,
    number_value: int | str,
) -> str:
    """Generate the prompt to determine the MAX_COST parameter.

    Args:
        profile: The user profile dictionary.
        variables_info: Descriptive string about profile variables.
        language_info: Descriptive string about supported languages.
        number_value: The previously determined 'number' parameter.

    Returns:
        The formatted prompt string for the LLM.
    """
    return f"""
    Determine the MAX_COST parameter for this conversation scenario.

    CONVERSATION SCENARIO: {profile.get("name", "Unnamed")}
    USER ROLE: {profile.get("role", "Unknown")}
    {variables_info}{language_info}

    PREVIOUSLY DETERMINED:
    NUMBER: {number_value}

    MAX_COST:
    - Set a budget limit for all conversations combined, in dollars.
    - For specific number conversations: typically 0.5-1.0 dollars.
    - For "all_combinations": use higher limits (1.5-3.0) since there will be more conversations.
    - For "sample(X)": scale based on X value - higher X needs higher budget.
    - IMPORTANT: Consider the complexity of the conversation and goals when setting this limit.

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    MAX_COST: number

    EXAMPLES:
    MAX_COST: 1.0
    MAX_COST: 2.0
    """


def get_goal_style_prompt(
    profile: dict[str, Any],
    variables_info: str,
    language_info: str,
    number_value: int | str,
    max_cost: float,
) -> str:
    """Generate the prompt to determine the GOAL_STYLE parameter.

    Args:
        profile: The user profile dictionary.
        variables_info: Descriptive string about profile variables.
        language_info: Descriptive string about supported languages.
        number_value: The previously determined 'number' parameter.
        max_cost: The previously determined 'max_cost' parameter.

    Returns:
        The formatted prompt string for the LLM.
    """
    return f"""
    Determine the GOAL_STYLE parameter for this conversation scenario.

    CONVERSATION SCENARIO: {profile.get("name", "Unnamed")}
    USER ROLE: {profile.get("role", "Unknown")}
    {variables_info}{language_info}

    PREVIOUSLY DETERMINED:
    NUMBER: {number_value}
    MAX_COST: {max_cost}

    GOAL_STYLE:
    Choose ONE option that best fits this conversation scenario:
    - "steps": Fixed number of conversation turns (e.g., 5 to 8).
    - "all_answered": Ends when all user goals are addressed, with a limit on total turns (e.g., 10 to 20).
    - "random_steps": Random number of turns up to a maximum value (e.g., 12). Useful to test conversations of different lengths, for example, very short interactions.


    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    GOAL_STYLE: type
    GOAL_STYLE_VALUE: value

    EXAMPLE OUTPUTS:
    GOAL_STYLE: steps
    GOAL_STYLE_VALUE: 6

    GOAL_STYLE: all_answered
    GOAL_STYLE_VALUE: 15
    """


def _format_goal_style_for_prompt(goal_style: dict[str, Any]) -> str:
    """Formats the goal_style dictionary into a string for the prompt."""
    if "steps" in goal_style:
        return f"{{'steps': {goal_style['steps']}}}"
    if "all_answered" in goal_style:
        limit = goal_style["all_answered"].get("limit", 15)
        return f"{{'all_answered': {{'limit': {limit}}}}}"
    if "random_steps" in goal_style:
        return f"{{'random_steps': {goal_style['random_steps']}}}"
    return str(goal_style)


def get_interaction_style_prompt(
    profile_context: PromptProfileContext,
    prev_params: PromptPreviousParams,
    lang_support: PromptLanguageSupport,
) -> str:
    """Generate the prompt to determine the INTERACTION_STYLE parameter.

    Args:
        profile_context: Dictionary containing profile, variables info, and language info.
        prev_params: Dictionary containing previously determined number, cost, and goal style.
        lang_support: Dictionary containing language support text and examples.

    Returns:
        The formatted prompt string for the LLM.
    """
    goal_style_str = _format_goal_style_for_prompt(prev_params["goal_style"])

    return f"""
    Determine the INTERACTION_STYLE parameter for this conversation scenario.

    CONVERSATION SCENARIO: {profile_context["profile"].get("name", "Unnamed")}
    USER ROLE: {profile_context["profile"].get("role", "Unknown")}
    {profile_context["variables_info"]}{profile_context["language_info"]}

    PREVIOUSLY DETERMINED:
    NUMBER: {prev_params["number_value"]}
    MAX_COST: {prev_params["max_cost"]}
    GOAL_STYLE: {goal_style_str}

    INTERACTION_STYLE:
    Choose 1-2 appropriate styles based on the user role and scenario:
    - "long phrases": user uses very long phrases to write any query
    - "change your mind": user changes their mind during conversation (good for ordering/booking)
    - "change language": user switches languages mid-conversation (requires list of languages)
    - "make spelling mistakes": user makes typos and spelling mistakes
    - "single question": user asks only one query per interaction
    - "all questions": user asks everything from goals in one interaction
    - "random": applies random styles from a specified list
    - If no style is specified, "default" is used (natural conversation style)

    RECOMMENDATION:
    - Select styles that match your user's likely behavior
    - Use "change your mind" for scenarios where selections are made
    - Use "single question" for complex conversations, "all questions" for simpler ones
    - Use "random" with a list when you want varied interaction styles
    - Use "change language" when the chatbot supports multiple languages {lang_support["supported_languages_text"]}

    EXAMPLES:
    - Single style: "make spelling mistakes"
    - Multiple styles: ["long phrases", "change your mind"]
    - Random with language change:
      random:
        - make spelling mistakes
        - all questions
        - long phrases
        - change language:
         {lang_support["languages_example"]}

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    INTERACTION_STYLE: style or [style1, style2]
    """
