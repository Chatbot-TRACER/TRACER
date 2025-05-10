"""Prompts for generating conversation parameters based on user profile information."""

from typing import Any, TypedDict

# Define simplified versions of functions locally to avoid circular imports
def _simple_get_profile_variables(profile: dict[str, Any]) -> list[str]:
    """Simple version of _get_profile_variables for the prompt template."""
    variables = []

    # Check for variables at the top level
    variables.extend([
        var_name
        for var_name, var_def in profile.items()
        if isinstance(var_def, dict) and "function" in var_def and "data" in var_def
    ])

    # Also check for variables nested within the 'goals' list
    if "goals" in profile and isinstance(profile["goals"], list):
        for item in profile["goals"]:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, dict) and "function" in value and "data" in value:
                        variables.append(key)

    return variables


def _simple_get_max_variable_size(profile: dict[str, Any]) -> int:
    """Simple version of _get_max_variable_size for the prompt template."""
    max_size = 1

    # Helper function to get variable definition
    def get_var_def(var_name):
        # Check top level
        var_def = profile.get(var_name)
        if isinstance(var_def, dict) and "function" in var_def and "data" in var_def:
            return var_def

        # Check in goals
        if "goals" in profile and isinstance(profile["goals"], list):
            for item in profile["goals"]:
                if isinstance(item, dict) and var_name in item:
                    var_def = item[var_name]
                    if isinstance(var_def, dict) and "function" in var_def and "data" in var_def:
                        return var_def
        return None

    # Check all variables
    for var_name in _simple_get_profile_variables(profile):
        var_def = get_var_def(var_name)
        if var_def and "data" in var_def:
            data = var_def.get("data", [])
            if isinstance(data, list):
                current_size = len(data)
                if current_size > max_size:
                    max_size = current_size

    return max_size


def _simple_calculate_combinations(profile: dict[str, Any], variables: list[str]) -> int:
    """Simple version of _calculate_combinations for the prompt template."""
    # Just count variables for a simplified estimate
    if not variables:
        return 1

    return max(len(variables), 2)

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
    # Get maximum forward variable length for calculation
    max_forward_size = 1
    for var_name, var_def in profile.items():
        if isinstance(var_def, dict) and "function" in var_def and "data" in var_def:
            data = var_def.get("data", [])
            if isinstance(data, list):
                current_size = len(data)
                if current_size > max_forward_size:
                    max_forward_size = current_size

    # For more detailed variable analysis - using simplified local versions
    variables = _simple_get_profile_variables(profile)
    max_var_size = _simple_get_max_variable_size(profile)
    potential_combinations = _simple_calculate_combinations(profile, variables)

    # Update recommendation based on actual calculations
    if max_var_size > max_forward_size:
        max_forward_size = max_var_size

    importance_notice = ""
    if max_forward_size > 3:
        importance_notice = f"\nIMPORTANT: For a profile with {len(variables)} variables where the largest has {max_forward_size} options, you MUST choose at least {max_forward_size} conversations to ensure adequate coverage!"

    if has_nested_forwards:
        number_section = f"""
        NUMBER:
        Choose ONE option:
        - Enter a fixed number matching the maximum variable size ({max_forward_size}) to cover possible combinations.
        - "all_combinations" to try all possible combinations (ONLY if there are fewer than 5 total combinations).
        - "sample(X)" where X is a decimal between 0.1 and 1.0 to test a fraction of the combinations.

        RECOMMENDATION:
        - For variables with forward dependencies, use at least {max_forward_size} conversations to ensure full coverage.
        - For nested forwards with {potential_combinations} potential combinations, consider "sample(0.2)" or a higher fixed number.{importance_notice}
        """
    else:
        number_section = f"""
        NUMBER:
        Choose a specific number based on the conversation complexity:
        - For simple conversations without variables: 2-3 runs
        - For conversations with variables: Use at least {max_forward_size} to ensure all possibilities are tested

        RECOMMENDATION:
        - This profile has {len(variables)} variables with a maximum of {max_forward_size} options.
        - To ensure adequate testing coverage, set the number to at least {max_forward_size}.{importance_notice}
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
    NUMBER: {max_forward_size}
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
