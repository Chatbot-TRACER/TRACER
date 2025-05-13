
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
Your task is to identify a MINIMAL and SUFFICIENT set of 'outputs' to extract from the chatbot's responses. These outputs verify that the chatbot correctly addressed the user's overall intent and specific goal-driven interactions, including those involving `{{variables}}`.

USER PROFILE:
Name: {profile_name}
Role: {profile_role}

USER GOALS (these will be executed, pay close attention to how `{{variables}}` are used):
{goals_and_vars_for_prompt_str}
{variable_definitions_for_prompt_str}

FUNCTIONALITIES ASSIGNED TO THIS PROFILE (these define what the chatbot can do):
{functionalities_str}

{language_instruction}

**Your Task: Define FEW, HIGH-IMPACT VERIFIABLE OUTPUTS**

1.  Review ALL user goals for this profile. Identify the **major pieces of information the chatbot is expected to provide** or the **key states it should confirm** as a result of the entire goal sequence. A single chatbot response (and thus a single output definition) might validate aspects of multiple user goals.
2.  **Focus on Critical Information & Confirmations:**
    *   What are the most important things the chatbot says that indicate success or provide crucial data? (e.g., a final order summary, a booking confirmation number, a list of requested options, an answer to a direct question).
    *   Define outputs for these critical points.
3.  **AGGRESSIVELY CONSOLIDATE:**
    *   If the chatbot provides a **summary or comprehensive confirmation** (e.g., an order summary detailing item type, size, quantity, and accessories after several interaction steps), define **ONE primary output** for that entire summary.
    *   **AVOID separate outputs for intermediate confirmations if they are encapsulated in a final, more complete confirmation.** For example, if the user selects an item, then a size, then a color, and the chatbot provides a final summary "You selected a large blue Gizmo", prioritize an output for the "final_gizmo_summary" rather than separate outputs for "item_confirmed", "size_confirmed", "color_confirmed".
    *   If one output field can capture the verification for multiple related goal aspects or variable iterations, PREFER THAT.
4.  **Output Definitions for Iterated Variables:**
    *   If a goal uses an iterated variable (e.g., `Request details for `{{service_id}}`), the single generic output field defined should capture the relevant chatbot response for the current iteration.
    *   **Naming Convention:** Generic (e.g., `service_id_response_details`).
    *   **Description for Extraction:** Clearly state WHAT to extract. Indicate it corresponds to the *specific value of the `{{variable_name}}`* used in that test instance. Provide extraction hints. DO NOT include `{{variable_name}}` in the DESCRIPTION field itself.
        *   GOOD Example: `output_name: item_specific_attribute_value`
            `DESCRIPTION: The chatbot's stated value for a specific attribute (e.g., color, material) of the item that corresponds to the current '{{item_id}}' variable used in the goal. Look for phrases like 'Attribute for [item_id_value]: [attribute_value]'.`
5.  **Data Types:** Assign ONE appropriate data type from: `int`, `float`, `money`, `str`, `string`, `time`, `date`.
6.  **Focus:** Outputs are information THE CHATBOT PROVIDES.
7.  **Minimize Outputs:** Generate the **absolute minimum number of output fields** required to comprehensively verify that the chatbot has successfully processed the user's goals and provided the necessary information, especially considering the final state or summary after a series of interactions.

**Output Format (Strictly follow this for EACH output):**
OUTPUT: output_name_1
TYPE: output_type_1
DESCRIPTION: A concise, static description of the information the chatbot is expected to provide.

OUTPUT: output_name_2
TYPE: output_type_2
DESCRIPTION: ...

Generate the necessary output definitions. Aim for the MOST CONSOLIDATED yet comprehensive set. Do NOT include any explanatory text before the first "OUTPUT:" line or after the last description.
"""
