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
You are an AI assistant designing test verification outputs for a chatbot user profile.
Your task is to identify GRANULAR and SPECIFIC outputs to extract from the chatbot's responses. These outputs should verify individual pieces of information and confirmations, allowing precise detection of what the chatbot might be missing or getting wrong.

USER PROFILE:
Name: {profile_name}
Role: {profile_role}

USER GOALS (these will be executed, pay close attention to how `{{variables}}` are used):
{goals_and_vars_for_prompt_str}
{variable_definitions_for_prompt_str}

FUNCTIONALITIES ASSIGNED TO THIS PROFILE (these define what the chatbot can do):
{functionalities_str}

{language_instruction}

**Your Task: Define GRANULAR, SPECIFIC VERIFIABLE OUTPUTS**

1. **Break Down Complex Information**: Instead of creating one output for "order_summary" or "appointment_confirmation", create separate outputs for each critical piece of information:
   - For orders: separate outputs for each item, quantity, price, total, order_id, delivery_date, etc.
   - For appointments: separate outputs for date, time, service_type, provider_name, location, etc.
   - For bookings: separate outputs for confirmation_number, check_in_date, check_out_date, room_type, guest_count, etc.

2. **Focus on Individual Data Points**: Each output should verify ONE specific piece of information that the chatbot should provide. This allows precise identification of missing or incorrect details.

3. **Consider All Goal Components**: Review each user goal and identify ALL the individual pieces of information the chatbot needs to confirm or provide throughout the interaction sequence.

4. **Variable-Based Outputs**: For goals with variables like `{{service_id}}` or `{{item_id}}`, create outputs that capture specific information about the current variable value being tested.

5. **Essential vs Optional Information**: Prioritize outputs for:
   - Required confirmations (dates, times, IDs, prices)
   - Critical details that indicate successful processing
   - Key information users need to verify their requests

6. **Data Types**: Assign appropriate data types from this exact list: `int`, `float`, `money`, `str`, `string`, `time`, `date`. Do NOT use any other types like 'boolean', 'bool', etc. For yes/no values, use `str` type.

7. **Naming Convention**: Use descriptive names that clearly indicate what specific information is being captured (e.g., `confirmed_appointment_date`, `order_total_price`, `selected_item_name`).

8. **IMPORTANT - Variable Placeholder Rule**: When writing output descriptions, do NOT include variable placeholders with curly braces like `{{variable_name}}`. Instead, refer to the concept generically. For example:
   - WRONG: "Confirms the selected item size ({{item_size}})"
   - CORRECT: "Confirms the selected item size"
   - WRONG: "The service ID ({{service_id}}) chosen by the user"
   - CORRECT: "The service ID chosen by the user"
   - WRONG: "Shows the booking date ({{booking_date}}) requested"
   - CORRECT: "Shows the booking date requested"

**Examples of Granular Outputs:**
- Instead of "booking_summary" → `reservation_confirmation_number`, `check_in_date`, `check_out_date`, `room_type`, `guest_count`, `total_price`
- Instead of "order_details" → `ordered_item_name`, `item_quantity`, `item_unit_price`, `order_total`, `estimated_delivery_date`, `order_confirmation_id`
- Instead of "appointment_info" → `appointment_date`, `appointment_time`, `service_type`, `provider_name`, `appointment_duration`

**Output Format (Strictly follow this for EACH output):**
OUTPUT: output_name_1
TYPE: output_type_1
DESCRIPTION: A concise description of the specific piece of information the chatbot should provide.

OUTPUT: output_name_2
TYPE: output_type_2
DESCRIPTION: ...

Generate comprehensive granular output definitions that allow verification of each critical piece of information separately. Do NOT include any explanatory text before the first "OUTPUT:" line or after the last description.
"""
