"""Prompts for grouping functionalities and generating profile goals."""

import re
from typing import Any, TypedDict

# --- Data Structures for Prompt Arguments ---


class ProfileGroupingContext(TypedDict):
    """Contextual information for grouping functionalities into profiles.

    Args:
        conversation_context: String describing the overall conversation history/context.
        workflow_context: String describing discovered conversation workflows.
        chatbot_type_context: String describing the type of chatbot (transactional/informational).
        language_instruction_grouping: Language-specific instructions for the grouping prompt.
    """

    conversation_context: str
    workflow_context: str
    chatbot_type_context: str
    language_instruction_grouping: str


class ProfileGoalContext(TypedDict):
    """Contextual information for generating goals for a specific profile.

    Args:
        chatbot_type_context: String describing the type of chatbot.
        workflow_context: String describing discovered conversation workflows.
        limitations: List of known chatbot limitations.
        conversation_context: String describing the overall conversation history/context.
        language_instruction_goals: Language-specific instructions for the goal generation prompt.
    """

    chatbot_type_context: str
    workflow_context: str
    limitations: list[str]
    conversation_context: str
    language_instruction_goals: str


# --- Language Instruction Functions ---


def get_language_instruction_grouping(primary_language: str) -> str:
    """Generate the language instruction for profile grouping."""
    if not primary_language:
        return ""
    return f"""
LANGUAGE REQUIREMENT:
- Write ALL profile names, roles, and functionalities in {primary_language}
- KEEP ONLY the formatting markers (##, PROFILE:, ROLE:, FUNCTIONALITIES:) in English
- Example if the primary language was Spanish:
  ## PROFILE: [Nombre del escenario en Spanish]
  ROLE: [Rol del usuario en Spanish]
  FUNCTIONALITIES:
  - [Funcionalidad en Spanish]
"""


def get_language_instruction_goals(primary_language: str) -> str:
    """Generate the language instruction for goal generation."""
    if not primary_language:
        return ""
    return f"""
LANGUAGE REQUIREMENT:
- Write ALL goals in {primary_language}
- KEEP ONLY the formatting markers (GOALS:) in English
- Keep variables in {{variable}} format
- Example in Spanish:
  GOALS:
  - "Primer objetivo en Spanish con {{variable}}"
"""


# --- Prompt Generation Functions ---


def get_profile_grouping_prompt(
    functionalities_with_details: list[str],
    context: ProfileGroupingContext,
) -> str:
    """Generates a prompt to group functionalities into logical user profiles."""
    functionality_list_str = "\n".join([f"- {f}" for f in functionalities_with_details])
    num_functionalities = len(functionalities_with_details)

    # Construct context section only if parts are available
    context_str = "\n--- CONTEXTUAL INFORMATION ---\n"
    has_context = False
    if context.get("chatbot_type_context"):
        context_str += context["chatbot_type_context"]
        has_context = True
    if context.get("workflow_context"):
        context_str += context["workflow_context"]
        has_context = True
    if context.get("conversation_context"):
        max_conv_len = 2000
        conv_snippet = context["conversation_context"]
        if len(conv_snippet) > max_conv_len:
            conv_snippet = conv_snippet[:max_conv_len] + "\n... (conversation truncated)"
        context_str += "\nSAMPLE CONVERSATIONS:\n" + conv_snippet + "\n"
        has_context = True

    if not has_context:
        context_str = ""
    else:
        context_str += "--- END CONTEXTUAL INFORMATION ---\n\n"

    # Add specific instructions based on chatbot_type
    grouping_strategy_instruction = ""
    if "TRANSACTIONAL" in context["chatbot_type_context"].upper():
        grouping_strategy_instruction = """
    **Grouping Strategy for TRANSACTIONAL Chatbot:**
    - Prioritize grouping functionalities that form a single, end-to-end user transaction or a significant sub-part of one (e.g., entire item ordering process, user registration, booking an appointment).
    - A single profile might cover an entire common transaction.
    - If multiple distinct transactions exist (e.g., 'make a purchase' vs. 'track an order'), create separate profiles for them.
    - Informational functionalities directly supporting a transaction (e.g., 'view item details' before 'add to cart') can be grouped within that transactional profile.
    """
    elif "INFORMATIONAL" in context["chatbot_type_context"].upper():
        grouping_strategy_instruction = """
    **Grouping Strategy for INFORMATIONAL Chatbot:**
    - Group functionalities that relate to a common theme, topic, or area of inquiry (e.g., all functionalities about 'account settings', all about 'product specifications for category X').
    - A single profile can cover multiple related questions a user might ask within one thematic session.
    - Aim for profiles that represent a user trying to get comprehensive information about a particular subject area.
    """

    return f"""
You are a User Profile Designer for chatbot testing. Your task is to group the following chatbot functionalities into a MINIMAL number of LOGICAL user profiles.
The primary goal is to ensure ALL functionalities are covered for testing across the generated profiles, using as FEW profiles as reasonably possible.

EXTREMELY IMPORTANT RESTRICTIONS (Apply to ALL profiles):
- Create ONLY profiles for realistic end users with PRACTICAL GOALS.
- NEVER create profiles about users asking about chatbot capabilities or limitations.
- NEVER create profiles where the user is trying to test or evaluate the chatbot.
- Focus ONLY on real-world user tasks and objectives.
- Profiles should be genuine use cases, not meta-conversations about the chatbot itself.

AVAILABLE CHATBOT FUNCTIONALITIES ({num_functionalities} total, including input/output details):
{functionality_list_str}
{context_str}
**Instructions for Grouping:**

1.  **Analyze Relationships & Workflow:** Examine the functionalities. Identify functionalities that are intrinsically related, often used together, or form a sequence in a user journey. The WORKFLOW INFORMATION (if provided) is CRITICAL for grouping sequential transactional steps.
2.  **Define Coherent Roles/Personas:** For each group of functionalities, define a user role/persona that would naturally use them together (e.g., "Prospective Student Inquiring about Admissions," "Customer Placing a Standard Order," "User Troubleshooting an Issue").
{grouping_strategy_instruction}
3.  **Logical Grouping for Coverage & Efficiency:** Assign functionalities to profiles to ensure:
    *   **Complete Coverage:** Every functionality from the input list MUST be assigned to at least one profile.
    *   **Logical Cohesion:** Functionalities within a profile should make sense for the defined role to perform in a related series of interactions.
    *   **Efficiency:** Aim for the minimum number of profiles needed to cover all functionalities logically. A profile can and often should cover multiple related functionalities. Avoid creating profiles for single, isolated functionalities if they can be logically grouped.
4.  **Describe the Role:** For each profile, provide a concise description of the user's role or primary objective.
5.  **Output Format:** Structure your response exactly as follows:

## PROFILE: [Profile Name (e.g., New Item Order and Customization)]
ROLE: [Description of the user role/objective for this profile]
FUNCTIONALITIES:
- [Full functionality description string as provided in input, assigned to this profile]
- [Another full functionality description string assigned to this profile]
...

{context["language_instruction_grouping"]}
Make sure profile names and roles are descriptive. Ensure ALL input functionalities are assigned. The goal is to create the SMALLEST number of profiles that logically cover all functionalities.
"""


def _extract_functionality_names(assigned_functionality_rich_strings: list[str]) -> list[str]:
    """Extract functionality names from rich strings.

    Args:
        assigned_functionality_rich_strings: List of functionality strings with metadata

    Returns:
        List of extracted functionality names or descriptions
    """
    assigned_func_names_or_descs_for_prompt = []
    for func_str in assigned_functionality_rich_strings:
        name_match = re.match(r"^([\w_]+):", func_str)
        desc_match = re.match(r"^(.*?)(?: \(Inputs:|\(Outputs:|$)", func_str)
        if name_match:
            assigned_func_names_or_descs_for_prompt.append(name_match.group(1))
        elif desc_match:
            assigned_func_names_or_descs_for_prompt.append(desc_match.group(1).strip())
        else:
            assigned_func_names_or_descs_for_prompt.append(func_str[:100] + "...")  # Truncated

    return assigned_func_names_or_descs_for_prompt


def _build_parameter_details(
    all_structured_functionalities: list[dict[str, Any]], profile_func_identifiers: list[str]
) -> tuple[str, bool]:
    """Build parameter details string for the prompt.

    Args:
        all_structured_functionalities: List of all functionality definitions
        profile_func_identifiers: List of functionality identifiers for this profile

    Returns:
        Tuple of (parameter_details_string, found_parameters_flag)
    """
    param_details_for_prompt = "\n\nPARAMETER DETAILS FOR TARGET FUNCTIONALITIES (use these EXACT parameter names as {{variable_name}} placeholders in goals):\n"
    found_params_for_profile = False

    for func_dict in all_structured_functionalities:
        func_name = func_dict.get("name")
        if func_name in profile_func_identifiers:
            params_in_func = []
            for p_dict in func_dict.get("parameters", []):
                if isinstance(p_dict, dict) and p_dict.get("name"):
                    param_name = p_dict.get("name")
                    param_options = p_dict.get("options", [])
                    param_info_str = f"'{param_name}'"
                    if param_options:
                        param_info_str += (
                            f" (known options: {', '.join(param_options[:3])}... use one of these or a placeholder)"
                        )
                    params_in_func.append(param_info_str)

            if params_in_func:
                param_details_for_prompt += f"- Functionality '{func_name}': Has parameters like {', '.join(params_in_func)}. Create goals that naturally use these with `{{{{parameter_name}}}}` placeholders.\n"
                found_params_for_profile = True

    if not found_params_for_profile:
        param_details_for_prompt = (
            "\n\nNo specific input parameters identified for target functionalities; focus goals on triggering them.\n"
        )

    return param_details_for_prompt, found_params_for_profile


def _build_output_info(all_structured_functionalities: list[dict[str, Any]]) -> tuple[str, bool]:
    """Build output information string for the prompt.

    Args:
        all_structured_functionalities: List of all functionality definitions

    Returns:
        Tuple of (output_info_string, found_output_flag)
    """
    output_as_param_info = "\n\nCONTEXTUAL KNOWLEDGE - CHATBOT CAN PROVIDE THESE LISTS OF CHOICES (These can be used as inputs for subsequent goal steps):\n"
    found_output_as_param = False

    for func_dict in all_structured_functionalities:
        func_name = func_dict.get("name")
        for output_spec in func_dict.get("outputs", []):
            if isinstance(output_spec, dict):
                output_category = output_spec.get("category")
                output_desc = output_spec.get("description", "")

                # Heuristic: if description contains commas or common list delimiters, it might be a list of options
                potential_options = []
                if output_desc and ("," in output_desc or ";" in output_desc or output_desc.count("\n") > 0):
                    # Simple split for now, can be refined
                    options_from_desc = [opt.strip() for opt in re.split(r"[,;\n]", output_desc) if opt.strip()]
                    if len(options_from_desc) > 1:  # If we found more than one item
                        potential_options = options_from_desc

                if potential_options:
                    output_as_param_info += f"- Functionality '{func_name}' can provide a list for '{output_category}' like: [{', '.join(potential_options[:3])}...]. A goal could be to first get this list, then use one of these values in a subsequent step.\n"
                    found_output_as_param = True

    if not found_output_as_param:
        output_as_param_info = "\n\n(No specific list-like outputs identified from other functionalities that could serve as choices for inputs.)\n"

    return output_as_param_info, found_output_as_param


def get_profile_goals_prompt(
    profile: dict[str, Any],
    all_structured_functionalities: list[dict[str, Any]],
    context: ProfileGoalContext,
) -> str:
    """Returns the prompt for generating user-centric goals for a profile.

    Args:
        profile: The specific profile dictionary (containing name, role, functionalities).
        all_structured_functionalities: A list of functionality objects related to the profile.
        context: A dictionary containing various contextual strings for the prompt.

    Returns:
        A formatted string representing the LLM prompt.
    """
    profile_name = profile.get("name", "Unnamed Profile")
    profile_role = profile.get("role", "User")

    # Functionalities assigned to THIS profile
    assigned_functionality_rich_strings = profile.get("functionalities", [])
    assigned_func_names_or_descs_for_prompt = _extract_functionality_names(assigned_functionality_rich_strings)

    functionalities_section = "TARGET FUNCTIONALITIES FOR THIS PROFILE (focus on testing these):\n" + "\n".join(
        [f"- {name_or_desc}" for name_or_desc in assigned_func_names_or_descs_for_prompt]
    )

    profile_func_identifiers = []
    for rich_string in assigned_functionality_rich_strings:
        temp_name = assigned_func_names_or_descs_for_prompt[assigned_functionality_rich_strings.index(rich_string)]
        profile_func_identifiers.append(temp_name)

    param_details_for_prompt, found_params_for_profile = _build_parameter_details(
        all_structured_functionalities, profile_func_identifiers
    )

    output_as_param_info, found_output_as_param = _build_output_info(all_structured_functionalities)

    return f"""
You are crafting a sequence of user goals for a specific test profile. These goals should guide a user simulator to test the assigned chatbot functionalities thoroughly, including their parameters and how they connect.

PROFILE NAME: {profile_name}
USER ROLE: {profile_role}
{context["chatbot_type_context"]}
{functionalities_section}
{param_details_for_prompt}
{output_as_param_info}
{context["workflow_context"]}
{context["conversation_context"]}
{context["language_instruction_goals"]}

**Goal Generation Instructions:**

1.  **Create 2-5 User-Centric Goals:** Goals must reflect what a real user wants to *achieve*.
2.  **Sequential & Coherent:** If testing a TRANSACTIONAL chatbot or a known workflow, goals should form a logical sequence of steps a user would take. Use the WORKFLOW INFORMATION and PARAMETER DETAILS.
3.  **Test Parameters with `{{variables}}`:** For functionalities with parameters (see PARAMETER DETAILS), formulate goals that naturally incorporate providing values for these parameters using their **exact names** as `{{parameter_name}}` placeholders.
4.  **Utilize Outputs as Inputs (Parameter Chaining):**
    Refer to "CONTEXTUAL KNOWLEDGE - CHATBOT CAN PROVIDE THESE LISTS OF CHOICES".
    If a functionality assigned to this profile *requires an input* that matches one of those listed choice categories (e.g., profile needs to test `select_item_type`, and you know `list_item_types` outputs "item_type_choices: TypeA, TypeB"), then:
    a. Create a goal to first *elicit* that list of choices from the chatbot (e.g., "Ask what item types are available.").
    b. Create a subsequent goal to *use one of those choices* (e.g., "Select the item type {{chosen_item_type}}.").
    The `{{chosen_item_type}}` variable will later be defined to iterate through TypeA, TypeB, etc.
5.  **Cover Assigned Functionalities:** The set of goals should aim to trigger the "TARGET FUNCTIONALITIES FOR THIS PROFILE".
6.  **Realistic & Practical:** Goals must be tasks a real user would perform.
7.  **Variable Placeholder Format:** Strictly use `{{variable_name}}`.
8.  **AVOID:**
    *   Goals about testing the chatbot, its limitations, or general knowledge.
    *   Meta-goals like "see what happens if I provide X."
    *   Goals that are too vague or too complex for a single user turn.

**Output Format:**

GOALS:
- "First user goal, possibly using {{variable_for_param1}}."
- "Second user goal, perhaps asking for choices, like 'What {{item_category_options}} are available?'"
- "Third goal, using a choice like 'Select the {{chosen_item_from_category}}'."

Generate ONLY the goals.
"""
