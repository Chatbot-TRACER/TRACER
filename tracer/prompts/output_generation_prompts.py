"""Module for generating output-related prompts for the TRACER system."""

from typing import Any

from tracer.constants import LIST_TRUNCATION_THRESHOLD


def _format_data_preview(var_def: dict[str, Any]) -> str:
    """Format variable data for preview display.

    Args:
        var_def: Variable definition dictionary containing data

    Returns:
        Formatted string preview of the variable data
    """
    data_preview = str(var_def.get("data", "N/A"))
    if isinstance(var_def.get("data"), list):
        actual_data_list = var_def.get("data", [])
        if len(actual_data_list) > LIST_TRUNCATION_THRESHOLD:
            data_preview = (
                f"{str(actual_data_list[:LIST_TRUNCATION_THRESHOLD])[:-1]}, ... (Total: {len(actual_data_list)} items)]"
            )
        else:
            data_preview = str(actual_data_list)
    elif isinstance(var_def.get("data"), dict):
        data = var_def["data"]
        data_preview = f"min: {data.get('min')}, max: {data.get('max')}, step: {data.get('step')}"

    return data_preview


def _process_profile_goals(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Process profile goals and extract string goals and variable details.

    Args:
        profile: Profile dictionary containing goals

    Returns:
        Tuple of (raw_string_goals, variable_details_list)
    """
    raw_string_goals = []
    variable_details_list = []

    for goal_item in profile.get("goals", []):
        if isinstance(goal_item, str):
            raw_string_goals.append(f"- {goal_item}")
        elif isinstance(goal_item, dict):
            for var_name, var_def in goal_item.items():
                if isinstance(var_def, dict):
                    data_preview = _format_data_preview(var_def)
                    variable_details_list.append(
                        f"  - Note: A variable '{{{var_name}}}' is used in goals, iterating with function '{var_def.get('function')}' using data like: {data_preview}."
                    )

    return raw_string_goals, variable_details_list


def _format_goals_and_variables(raw_string_goals: list[str], variable_details_list: list[str]) -> tuple[str, str]:
    """Format goals and variables for prompt display.

    Args:
        raw_string_goals: List of string goals
        variable_details_list: List of variable details

    Returns:
        Tuple of (goals_string, variable_definitions_string)
    """
    goals_and_vars_for_prompt_str = "\\n".join(raw_string_goals)
    variable_definitions_for_prompt_str = ""

    if variable_details_list:
        variable_definitions_for_prompt_str = (
            "\\n\\nIMPORTANT VARIABLE CONTEXT (variables like `{{variable_name}}` in goals will iterate through values like these):\\n"
            + "\\n".join(variable_details_list)
        )

    if not raw_string_goals and not variable_details_list:
        goals_and_vars_for_prompt_str = "- (No specific string goals or variables with options defined. Define generic outputs based on role and functionalities.)\\n"
    elif not raw_string_goals and variable_details_list:
        goals_and_vars_for_prompt_str = (
            "- (Primary interaction driven by variable iterations. Define outputs to verify these.)\\n"
        )

    return goals_and_vars_for_prompt_str, variable_definitions_for_prompt_str


def get_outputs_prompt(
    profile: dict[str, Any],
    profile_functionality_details: list[str],
    language_instruction: str,
) -> str:
    """Generate a prompt for creating outputs for a chatbot profile.

    Args:
        profile: The profile dictionary containing goals and other profile information
        profile_functionality_details: List of functionality details for the profile
        language_instruction: Instructions for the language to use in outputs

    Returns:
        A formatted string prompt for output generation
    """
    profile_name = profile.get("name", "Unnamed Profile")
    profile_role = profile.get("role", "Unknown Role")

    raw_string_goals, variable_details_list = _process_profile_goals(profile)
    goals_and_vars_for_prompt_str, variable_definitions_for_prompt_str = _format_goals_and_variables(
        raw_string_goals, variable_details_list
    )

    functionalities_str = "\\n".join([f"- {f_desc_str}" for f_desc_str in profile_functionality_details])

    return f"""
You are designing test outputs to verify what information a chatbot extracts during conversations.

USER PROFILE:
Name: {profile_name}
Role: {profile_role}

USER GOALS:
{goals_and_vars_for_prompt_str}
{variable_definitions_for_prompt_str}

CHATBOT FUNCTIONALITIES:
{functionalities_str}

{language_instruction}

**TASK: Define granular, verifiable outputs**

Break down each interaction into individual data points the chatbot should capture. Each output verifies ONE specific piece of information.

**TYPE SELECTION GUIDE:**
- `string`: IDs, codes, names, addresses, descriptions, status values
- `int`: Whole number counts (people, items when fractional impossible)
- `float`: Quantities that can be decimal (weight, duration, ratings, measurements)
- `money`: Prices, costs, totals, fees
- `date`: Dates (YYYY-MM-DD format)
- `time`: Times (HH:MM format)

**DESCRIPTION RULES:**
- Keep descriptions SHORT (3-8 words)
- Use format: "[noun] of the [context]" or "[what it represents]"
- Be specific and direct
- No variable placeholders like {{variable_name}}

**GOOD EXAMPLES:**
OUTPUT: transaction_id
TYPE: string
DESCRIPTION: Unique identifier for the transaction

OUTPUT: start_time
TYPE: time
DESCRIPTION: Scheduled start time

OUTPUT: item_quantity
TYPE: float
DESCRIPTION: Number of items requested

OUTPUT: processing_fee
TYPE: money
DESCRIPTION: Administrative processing cost

OUTPUT: scheduled_date
TYPE: date
DESCRIPTION: Date for the scheduled event

**BAD EXAMPLES:**
- "A comprehensive description detailing the transaction reference number assigned by the system"
- "The total duration ({{service_time}}) selected by the customer for their booking"
- Using `int` for reference codes or `string` for numeric measurements

**COVERAGE CHECKLIST:**
- Identifiers (order IDs, confirmation numbers, reference codes)
- Quantities and measurements (counts, weights, durations)
- Financial information (prices, totals, fees, discounts)
- Temporal data (dates, times, deadlines)
- Selections (chosen options, preferences, specifications)
- Confirmations (status updates, verifications)

**OUTPUT FORMAT:**
OUTPUT: output_name
TYPE: data_type
DESCRIPTION: Brief description

Generate comprehensive outputs covering all critical information points. Start immediately with the first "OUTPUT:" line.
"""
