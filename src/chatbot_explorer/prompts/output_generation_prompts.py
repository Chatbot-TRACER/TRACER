
import re
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
                    data_preview = str(var_def.get('data', 'N/A'))
                    if isinstance(var_def.get('data'), list):
                        actual_data_list = var_def.get('data', [])
                        if len(actual_data_list) > 3:
                            data_preview = f"{str(actual_data_list[:3])[:-1]}, ... (Total: {len(actual_data_list)} items)]"
                        else:
                            data_preview = str(actual_data_list)
                    elif isinstance(var_def.get('data'), dict):
                        data_preview = f"min: {var_def['data'].get('min')}, max: {var_def['data'].get('max')}, step: {var_def['data'].get('step')}"

                    variable_details_list.append(
                        f"  - Note: A variable '{{{var_name}}}' is used in goals, iterating with function '{var_def.get('function')}' using data like: {data_preview}."
                    )

    goals_and_vars_for_prompt_str = "\\n".join(raw_string_goals)
    if variable_details_list:
        variable_definitions_for_prompt_str = (
            "\\n\\nIMPORTANT VARIABLE CONTEXT (variables like `{{variable_name}}` in goals will iterate through values like these):\\n" +
            "\\n".join(variable_details_list)
        )

    if not raw_string_goals and not variable_details_list :
        goals_and_vars_for_prompt_str = "- (No specific string goals or variables with options defined. Define generic outputs based on role and functionalities.)\\n"
    elif not raw_string_goals and variable_details_list:
        goals_and_vars_for_prompt_str = "- (Primary interaction driven by variable iterations. Define outputs to verify these.)\\n"

    functionalities_str = "\\n".join([f"- {f_desc_str}" for f_desc_str in profile_functionality_details])

    return f"""
You are an AI assistant designing test verification outputs for a chatbot user profile.
For the given user profile, identify specific pieces of information that the CHATBOT IS EXPECTED TO PROVIDE in its responses. These 'outputs' will be extracted to validate its correctness during test execution.

USER PROFILE:
Name: {profile_name}
Role: {profile_role}

USER GOALS (these will be executed, pay close attention to how `{{variables}}` are used):
{goals_and_vars_for_prompt_str}
{variable_definitions_for_prompt_str}

FUNCTIONALITIES ASSIGNED TO THIS PROFILE (these define what the chatbot can do):
{functionalities_str}

{language_instruction}

**Your Task: Define VERIFIABLE OUTPUTS Based on Goals and Variables**

1.  **Identify Key Verifiable Information:** For each goal, what specific data should the chatbot provide that confirms the goal was addressed?
2.  **Output Definitions for Iterated Variables (CRITICAL):**
    *   If a goal uses a variable that iterates through a list (e.g., `Request details for `{{service_id}}`), define **one single, generic output field**.
    *   **Naming Convention:** Name the output generically (e.g., `service_id_response_details`).
    *   **Description for Variables:** The description should define *what piece of information the chatbot provides* in response to the goal using the variable. It should be understandable that this information will correspond to the specific value of the variable used in that test instance. **Avoid meta-descriptions about the testing process itself (e.g., do not say 'This output captures data for each iteration').** Focus on describing the *chatbot's output content*.
        *   GOOD Example for a goal like "Get configuration guide for `{{device_os}}`" (where `{{device_os}}` iterates through 'Android', 'iOS', 'Windows'):
            `OUTPUT: device_configuration_guide`
            `TYPE: str`
            `DESCRIPTION: The configuration guide (e.g., a URL or specific instructions) provided by the chatbot for the specified device operating system.` (Implies it's for the current iterated `{{device_os}}`)
        *   GOOD Example for a goal like "Order one `{{item_choice}}`":
            `OUTPUT: item_order_confirmation_message`
            `TYPE: str`
            `DESCRIPTION: The chatbot's confirmation message received after ordering the selected item, including details of that item.` (Implies the item details will match `{{item_choice}}`)
3.  **Specific Outputs for Non-Variable Goals:** Define outputs capturing specific info (e.g., if goal is "Ask for general support hours", output could be `general_support_hours_info`).
4.  **Data Types:** Assign ONE appropriate data type from: `int`, `float`, `money`, `str`, `string`, `time`, `date`.
5.  **Focus:** Outputs MUST be information THE CHATBOT PROVIDES.
6.  **Number of Outputs:** Generate outputs as needed to verify key information from goals. One well-described generic output can cover many variable iterations. Prioritize consolidation.

**Output Format (Strictly follow this for EACH output):**
OUTPUT: output_name_1
TYPE: output_type_1
DESCRIPTION: A concise description of the information the chatbot is expected to provide in relation to the user's goal.

OUTPUT: output_name_2
TYPE: output_type_2
DESCRIPTION: ...

**Example for a variable-driven goal:**
Goal: "Retrieve status for `{{order_reference}}`."
Variable `{{order_reference}}` data might iterate through ["REF001", "REF002", "REF003"].

Potential Output:
OUTPUT: order_status_update
TYPE: str
DESCRIPTION: The status update message provided by the chatbot for the specified order reference.

Generate the necessary output definitions. Aim for a concise yet comprehensive set. Do NOT include any explanatory text before the first "OUTPUT:" line or after the last description.
"""
