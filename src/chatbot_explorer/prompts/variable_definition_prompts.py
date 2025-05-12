"""Prompts for defining variables in user simulator profiles."""

from typing import Any, TypedDict


class ProfileContext(TypedDict):
    """Context information about a profile for variable definition.

    Attributes:
        name: Profile name
        role: User role description
        goals_text: Multi-line string containing profile goals
    """

    name: str
    role: str
    goals_text: str


def get_variable_definition_prompt(
    profile_context: ProfileContext,
    variable_name: str,
    all_other_variables: list[str],
    language_instruction: str,
) -> str:
    """Creates a text prompt for LLM to define a variable's parameters.

    Args:
        profile_context: Profile information (name, role, goals)
        variable_name: The variable name to define
        all_other_variables: Other variable names in the same profile
        language_instruction: Language-specific generation instructions

    Returns:
        Formatted prompt string for LLM to define the variable
    """
    other_vars_text = (
        f"OTHER VARIABLES IN THIS PROFILE: {', '.join(sorted(all_other_variables))}" if all_other_variables else ""
    )

    return f"""
    Define parameters for the variable '{variable_name}' used in a user simulator profile.

    USER PROFILE CONTEXT:
    Name: {profile_context["name"]}
    Role: {profile_context["role"]}
    Goals (where the variable might appear):
    {profile_context["goals_text"]}
    {other_vars_text} # Provides context for potential forward(other_var) usage.

    {language_instruction}

    Define ONLY the parameters for the variable '{variable_name}' following these guidelines:

    1. Choose ONE appropriate FUNCTION from:
        - default(): assigns ALL data in the list to the variable
        - random(): picks ONE random sample from the list
        - random(X): picks X random samples where X is LESS THAN the total number of items
        - random(rand): picks a random number of random samples
        - another(): picks different samples without repetition each time
        - another(X): picks X different samples without repetition
        - forward(): iterates through each sample one by one
        - forward(other_var): iterates and nests with other_var

    IMPORTANT FUNCTION RESTRICTIONS:
    - DO NOT nest functions (e.g., random(another(3)) is INVALID)
    - DO NOT use random(X) where X equals the total number of items (use default() instead)
    - INT() is NOT a function but a TYPE
    - Use forward() for systematic iteration through all values in order
    - Use random() for picking just one value each time (but there could be repetition)
    - Use another() when you want different values on subsequent uses

    2. Choose the most appropriate TYPE from:
        - string: for text values
        - int: for whole numbers
        - float: for decimal numbers

    3. IMPORTANT: Provide DATA in the correct format:
        - For string variables: use a list of realistic string values, the length of the list should be at least 3 items and at most 10 items, use the amount of data that makes sense for the variable
        - For int variables: ALWAYS use the min/max/step format like this:
            min: 1
            max: 10
            step: 1
        - For float variables: ALWAYS use the min/max/step format OR linspace like this:
            min: 1.0
            max: 5.0
            step: 0.5
            OR
            min: 1.0
            max: 5.0
            linspace: 5

    FORMAT YOUR RESPONSE FOR STRING VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: string
        DATA:
        - value1
        - value2
        - value3

        FORMAT YOUR RESPONSE FOR NUMERIC VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: int/float
        DATA:
        min: 1
        max: 10
        step: 1

    Respond ONLY with the FUNCTION, TYPE, and DATA for the variable '{variable_name}'.
    """


def get_variable_to_datasource_matching_prompt(
    goal_variable_names: list[str],
    potential_data_sources: list[dict[str, Any]],
    profile_role_and_goals: str,
) -> str:
    """Generates a prompt for the LLM to match goal variables to potential data sources
    (which can be direct parameters or list-like outputs from functionalities).
    """
    sources_str_list = []
    for i, source in enumerate(potential_data_sources):
        full_options = source.get("options", [])
        options_preview_str = "N/A"
        if isinstance(full_options, list) and full_options:
            options_preview_str = (
                f"{full_options[:3]}{'...' if len(full_options) > 3 else ''} (Total: {len(full_options)} options)"
            )

        sources_str_list.append(
            f"  {i + 1}. ID: DS{i + 1} ## Source Name: '{source.get('source_name')}'\n"
            f"     Type: {source.get('type', 'unknown')}\n"
            f"     Description: {source.get('source_description', 'N/A')}\n"
            f"     Example Options Preview: {options_preview_str}\n"
            f"     Origin Functionality: '{source.get('origin_func')}'"
        )
    sources_formatted_str = "\n".join(sources_str_list)

    variables_formatted_str = "\n".join([f"- '{var}'" for var in goal_variable_names])

    return f"""
You are an AI assistant helping to define test data for chatbot user profile goals.
Your task is to match variables found in user goals with the most appropriate data source from a provided list.
A data source can be a direct input parameter of a chatbot function OR an output of a chatbot function that provides a list of choices.

PROFILE CONTEXT (Role and Goals):
{profile_role_and_goals}

GOAL VARIABLES TO MATCH:
{variables_formatted_str}

AVAILABLE DATA SOURCES (Parameters or List-like Outputs):
{sources_formatted_str}

**MATCHING INSTRUCTIONS:**
1.  For each 'GOAL VARIABLE', find the BEST 'DATA SOURCE' from the list above.
2.  A match is good if the variable's intended meaning aligns with the data source's name, description, and example options.
3.  **TYPE COMPATIBILITY: Crucially, the *nature* of the GOAL VARIABLE must align with the *nature* of the options in the DATA SOURCE. For example, if a GOAL VARIABLE clearly implies a numeric input (e.g., '{{number_of_items}}', '{{age}}'), DO NOT match it to a DATA SOURCE that provides a list of textual names or categories (e.g., item types, colors) unless the variable name *explicitly* suggests choosing from those named categories.**
4.  **Prioritize sources of type 'direct_parameter' if the variable name closely matches the source_name of a direct parameter.**
5.  Consider 'output_as_parameter_options' when a goal variable seems to represent a *choice to be made from a list that the chatbot would provide*. The variable name might be more generic (e.g., `{{chosen_item}}`) while the data source name is more specific (e.g., `item_type_options`).
6.  If a variable clearly relates to a specific "Origin Functionality" mentioned in a data source, that increases match likelihood.
7.  If multiple data sources seem plausible, choose the one whose `source_name` or `source_description` is most semantically similar to the GOAL VARIABLE.
8.  **Crucially, the `options` from the matched DATA SOURCE will be used as the data for the GOAL VARIABLE.**

Output your matches as a JSON object where keys are the **exact** 'GOAL VARIABLE' names and values are objects containing:
- "matched_data_source_id": "The ID of the matched Data Source (e.g., DS1, DS2)".

Example JSON Output:
{{
  "{{variable_name_from_goal1}}": {{
    "matched_data_source_id": "DS3"
  }},
  "{{variable_name_from_goal2}}": {{
    "matched_data_source_id": "DS1"
  }}
}}

If a GOAL VARIABLE has no good match in the AVAILABLE DATA SOURCES, **DO NOT include it in your JSON output.**
**It is critical that you only provide a match if the options from the data source are semantically appropriate for the variable. For example, do not match a variable like 'number_of_items' to a data source that lists item types.**
Return ONLY the JSON object, which might be empty if no good matches are found.
"""


def get_clean_and_suggest_negative_option_prompt(
    dirty_options: list[str],
    variable_name: str,
    profile_goals_context: str,
    language: str,
) -> str:
    options_str = "\n".join([f"- {opt}" for opt in dirty_options])

    return f"""
You are an AI assistant helping prepare test data for a chatbot.
Your task is to process a list of "dirty" options for a variable named '{variable_name}'. These options were extracted from chatbot messages and may contain extra descriptive text.

This variable '{variable_name}' is used in user goals like:
{profile_goals_context[:300]}...

Language of options: {language}

"Dirty" Options List for '{variable_name}':
{options_str}

**Your Tasks:**

1.  **Clean the Options:**
    *   For each item in the "Dirty" Options List, extract only the core, usable option name or value.
    *   Remove any surrounding explanatory text (e.g., "Price for", "in small sizes", "option for").
    *   If an option becomes empty after cleaning, omit it from the cleaned list.
    *   Try to preserve original casing if it seems important (e.g., proper nouns like "Coke", "Margarita").
    *   Present the cleaned options as a list, each item starting with '- '.

2.  **Suggest One Out-of-Scope/Invalid Option:**
    *   Based on the cleaned valid options and the goals, suggest ONE plausible but LIKELY UNSUPPORTED or INVALID option for the variable '{variable_name}'.
    *   This invalid option should be something a user might mistakenly ask for related to the variable's purpose (e.g., if valid options are types of {variable_name}, suggest another related but unsupported type of {variable_name}).
    *   Avoid generic gibberish. The invalid option should make sense in the context but not be in the valid list.
    *   If you cannot confidently suggest a distinct invalid option, output "None".

**Output Format (Strictly follow this):**

CLEANED_OPTIONS:
- [Cleaned Option 1]
- [Cleaned Option 2]
...

INVALID_OPTION_SUGGESTION:
[Your single suggested invalid option OR the word "None"]

**Example:**
Dirty Options List for 'drink_type':
- Price for Coke
- Sprite is also available
- The Water option

CLEANED_OPTIONS:
- Coke
- Sprite
- Water

INVALID_OPTION_SUGGESTION:
Fanta
---
Dirty Options List for 'item_size':
- Small size (S)
- The medium option
- Large (L)

CLEANED_OPTIONS:
- Small
- Medium
- Large

INVALID_OPTION_SUGGESTION:
Extra Large
"""
