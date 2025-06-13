"""Prompts for defining variables in user simulator profiles."""

from typing import Any, TypedDict

from tracer.constants import VARIABLE_PATTERNS


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
        FUNCTION: function_name()
        TYPE: string
        DATA:
        - value1
        - value2
        - value3

        FORMAT YOUR RESPONSE FOR NUMERIC VARIABLES AS:
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
    """Generates a prompt for the LLM to match ONE goal variable to potential data sources."""
    sources_str_list = []
    for i, source in enumerate(potential_data_sources):
        full_options = source.get("options", [])
        options_preview_str = "N/A"
        if isinstance(full_options, list) and full_options:
            # Show more option examples to help with semantic matching
            preview_count = min(5, len(full_options))
            options_preview_str = f"{full_options[:preview_count]}{'...' if len(full_options) > preview_count else ''} (Total: {len(full_options)} options)"

        sources_str_list.append(
            f"  {i + 1}. ID: DS{i + 1} ## Source Name: '{source.get('source_name')}'\n"
            f"     Type: {source.get('type', 'unknown')}\n"
            f"     Description: {source.get('source_description', 'N/A')}\n"
            f"     Example Options Preview: {options_preview_str}\n"
            f"     Origin Functionality: '{source.get('origin_func')}'"
        )
    sources_formatted_str = "\n".join(sources_str_list)

    # Since we're matching one variable at a time, simplify
    variable_name = goal_variable_names[0] if goal_variable_names else "unknown"

    return f"""
You are an AI assistant helping to match a chatbot variable to appropriate data sources.

VARIABLE TO MATCH: '{variable_name}'

PROFILE CONTEXT: {profile_role_and_goals}

AVAILABLE DATA SOURCES:
{sources_formatted_str}

**MATCHING INSTRUCTIONS:**

1. **UNDERSTAND THE VARIABLE**: '{variable_name}' represents what type of content/value a user would provide.

2. **EXAMINE DATA SOURCES**: Look at the "Example Options Preview" - do these contain realistic values that users would actually input for '{variable_name}'?

3. **SEMANTIC MATCHING PRINCIPLES**:
   - Match based on what USERS would realistically provide, not system descriptions
   - Focus on actual content/values, not meta-information about the variable
   - Consider the user's perspective: "What would I say/select for '{variable_name}'?"

4. **MATCHING EXAMPLES**:
   - "question" variable → sources with actual questions users ask ("How much does it cost?", "What are your hours?")
   - "service_type" variable → sources with actual service names ("Basic cleaning", "Deep repair")
   - "date" variable → sources with actual dates ("Tomorrow", "Monday", "January 15")
   - "problem_description" variable → sources with actual problem descriptions users give

5. **AVOID MATCHING TO**:
   - Meta-descriptions: "The question", "Date selection", "Service information"
   - System references: "Variable options", "Data fields", "Category selection"
   - Incomplete fragments: "including", "such as", partial sentences

**CRITICAL QUESTION**: "Would the options in this data source contain realistic values that users would actually provide for '{variable_name}'?"

**YOUR TASK**:
- If you find a data source whose options contain realistic user values for '{variable_name}', respond with ONLY the ID (e.g., "DS3")
- If NO data source contains appropriate user values for '{variable_name}', respond with ONLY "NO_MATCH"

**IMPORTANT**: Prioritize user-realistic content over system descriptions. It's better to respond "NO_MATCH" than match to inappropriate options.

Your response (ID or NO_MATCH):"""


def get_clean_and_suggest_negative_option_prompt(
    dirty_options: list[str],
    variable_name: str,
    profile_goals_context: str,
    language: str,
) -> str:
    """Generate prompt to clean and suggest negative options for a variable.

    Args:
        dirty_options: List of raw options that need cleaning
        variable_name: Name of the variable these options belong to
        profile_goals_context: Context about the profile goals
        language: Language for the response

    Returns:
        Formatted prompt string for cleaning and suggesting negative options
    """
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


def get_variable_semantic_validation_prompt(
    variable_name: str,
    options: list[str],
) -> str:
    """Creates a prompt to validate if options are semantically appropriate for a variable.

    Args:
        variable_name: The variable name without {{ }}
        options: List of options to validate

    Returns:
        The prompt for semantic validation
    """
    options_str = "\n".join([f"- {opt}" for opt in options])
    variable_name_lower = variable_name.lower()

    # Look for common patterns in variable name using centralized patterns
    is_date_var = any(date_term in variable_name_lower for date_term in VARIABLE_PATTERNS["date"])
    is_time_var = any(time_term in variable_name_lower for time_term in VARIABLE_PATTERNS["time"])
    is_type_var = any(type_term in variable_name_lower for type_term in VARIABLE_PATTERNS["type"])
    is_number_of_var = any(number_of_term in variable_name_lower for number_of_term in VARIABLE_PATTERNS["number_of"])

    # Create guidance based on variable name patterns - language agnostic approach
    specific_guidance = ""
    if is_date_var:
        specific_guidance = "This variable represents a DATE. Options should be actual dates a user would select (like 'Tomorrow', 'Monday', '2024-01-15'), NOT meta-descriptions like 'The date' or 'Date selection'."
    elif is_time_var:
        specific_guidance = "This variable represents a TIME. Options should be actual times a user would select (like '9:00 AM', '14:30', 'Morning'), NOT meta-descriptions like 'The time' or 'Time selection'."
    elif is_type_var:
        base_term = variable_name_lower
        for type_marker in VARIABLE_PATTERNS["type"]:
            if type_marker in base_term:
                base_term = base_term.replace(type_marker, "").strip("_")
                break

        if base_term:
            specific_guidance = f"This variable represents TYPES or CATEGORIES of {base_term}. Options should be names of specific {base_term} types, NOT meta-descriptions about the variable itself."
    elif is_number_of_var:
        specific_guidance = "This variable represents a COUNT or QUANTITY. Options should be actual counts a user would select (like '1', '2', '5', 'One'), NOT meta-descriptions like 'The quantity'."

    return f"""
You are evaluating data quality for a chatbot test variable.

VARIABLE NAME: {variable_name}

CANDIDATE OPTIONS:
{options_str}

{specific_guidance}

**YOUR TASK:**
Determine if these options are semantically appropriate for the variable '{variable_name}'.

**EVALUATION CRITERIA:**

✓ GOOD Options:
- Actual values/names a user would realistically select or mention
- Specific instances of what the variable represents
- Real-world options that make sense in context

Examples: For "contact_method" → "Email", "Phone", "Chat" ✓
Examples: For "movie_type" → "Action", "Comedy", "Drama" ✓

✗ BAD Options:
- Meta-descriptions about the variable itself
- System references or administrative text
- Incomplete sentence fragments
- Category descriptions instead of actual values

Examples: For any variable → "The [variable]", "Available [options]" ✗
Examples: For any variable → "Selection menu", "including", "such as" ✗

**KEY QUESTION:** Are these realistic options that a real user would actually choose or mention when providing a '{variable_name}'?

**IMPORTANT:** Be permissive with real-world content. Focus on rejecting only obvious meta-descriptions, not legitimate option names.

Let's think step by step.

After your reasoning, answer with ONLY "Yes" or "No":
- "Yes" if ALL options are realistic user-selectable values for '{variable_name}'
- "No" if ANY options are meta-descriptions or inappropriate content
"""
