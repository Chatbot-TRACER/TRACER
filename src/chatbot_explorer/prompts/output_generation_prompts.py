from typing import Any


def get_outputs_prompt(
    profile: dict[str, Any],
    profile_functionality_details: list[str],
    language_instruction: str,
) -> str:
    profile_name = profile.get("name", "Unnamed Profile")
    profile_role = profile.get("role", "Unknown Role")

    goals_and_vars_for_prompt_str = ""
    variable_definitions_for_prompt_str = ""
    raw_string_goals = []
    variable_details_list = []

    for goal_item in profile.get("goals", []):
        if isinstance(goal_item, str):
            raw_string_goals.append(f"- {goal_item}")
        elif isinstance(goal_item, dict):
            for var_name, var_def in goal_item.items():
                if isinstance(var_def, dict):
                    data_preview = str(var_def.get("data", "N/A"))
                    if isinstance(var_def.get("data"), list):
                        actual_data_list = var_def.get("data", [])
                        if len(actual_data_list) > 3:
                            data_preview = (
                                f"{str(actual_data_list[:3])[:-1]}, ... (Total: {len(actual_data_list)} items)]"
                            )
                        else:
                            data_preview = str(actual_data_list)
                    elif isinstance(var_def.get("data"), dict):
                        data_preview = f"min: {var_def['data'].get('min')}, max: {var_def['data'].get('max')}, step: {var_def['data'].get('step')}"

                    variable_details_list.append(
                        f"  - Note: A variable '{{{var_name}}}' is used in goals, iterating with function '{var_def.get('function')}' using data like: {data_preview}."
                    )

    goals_and_vars_for_prompt_str = "\\n".join(raw_string_goals)
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
