"""Prompts for refining functionality nodes based on existing descriptions."""

import json
from typing import Any

from tracer.schemas.functionality_node_model import FunctionalityNode


def get_duplicate_check_prompt(node: FunctionalityNode, existing_descriptions: list[str]) -> str:
    """Generate prompt to check if a node is a duplicate of existing nodes."""
    return f"""
    Determine if the new functionality is semantically equivalent to any existing functionality.

    NEW FUNCTIONALITY:
    Name: {node.name}
    Description: {node.description}

    EXISTING FUNCTIONALITIES:
    {json.dumps(existing_descriptions, indent=2)}

    A functionality is a duplicate if it represents the SAME ACTION/CAPABILITY, even if described differently.

    IMPORTANT: You MUST respond in one of these two exact formats:
    1. If the functionality is unique:
       UNIQUE
       [brief explanation]

    2. If it is a duplicate, you MUST use the EXACT node name from the list provided:
       DUPLICATE_OF: [exact_name_from_list]
       [brief explanation]

    DO NOT use generic terms like 'ANY' or create new names. Only use names that exactly match one from the existing functionalities list.
    """


def get_relationship_validation_prompt(parent_node: FunctionalityNode, child_node: FunctionalityNode) -> str:
    """Generate prompt to validate parent-child relationship between nodes."""
    return f"""
    Evaluate if the 'POTENTIAL SUB-FUNCTIONALITY' represents a logical next step or a sub-component within a workflow that includes the 'PARENT FUNCTIONALITY'.

    PARENT FUNCTIONALITY:
    Name: {parent_node.name}
    Description: {parent_node.description}

    POTENTIAL SUB-FUNCTIONALITY (CHILD):
    Name: {child_node.name}
    Description: {child_node.description}

    Consider these criteria for a VALID relationship:
    1.  **Direct Sub-Task:** The child is a specific part needed to complete the parent's broader goal (e.g., Parent: `create_account`, Child: `set_password`).
    2.  **Refinement/Specification:** The child provides more detail or options related to the parent (e.g., Parent: `search_items`, Child: `apply_filter_to_results`).
    3.  **Sequential Step:** The child is a common or necessary step that logically occurs *after* the parent functionality is performed in a typical user workflow (e.g., Parent: `select_item`, Child: `add_item_to_cart`; Parent: `add_item_to_cart`, Child: `proceed_to_checkout`).
    4.  **Converging Path:** The child represents a common step that might be reached after *multiple different* preceding functionalities (like the parent). For example, `proceed_to_checkout` could validly follow both `select_standard_item` and `configure_custom_item`. If the child is a plausible *next step* after the parent, the relationship can be VALID even if other paths also lead to the child.

    EXAMPLE VALID RELATIONSHIPS:
    - Parent: `search_products`, Child: `filter_search_results` (Refinement)
    - Parent: `schedule_appointment`, Child: `confirm_appointment_details` (Sub-Task/Sequential)
    - Parent: `select_delivery_method`, Child: `provide_delivery_address` (Sequential)
    - Parent: `add_main_item_to_order`, Child: `offer_side_items` (Sequential/Enhancement)
    - Parent: `offer_side_items`, Child: `proceed_to_payment` (Sequential/Convergence) # Example: Sides might lead to payment
    - Parent: `configure_custom_product`, Child: `proceed_to_payment` (Sequential/Convergence) # Example: Custom config might also lead to payment

    EXAMPLE INVALID RELATIONSHIPS:
    - Parent: `login`, Child: `view_product_catalog` (Related but not typically a direct sequential step initiated by the login action itself)
    - Parent: `check_weather`, Child: `translate_text` (Unrelated domains)
    - Parent: `provide_order_summary`, Child: `search_products` (Wrong sequence)

    Focus on typical process flow and logical dependency. Does completing the PARENT naturally lead into needing or performing the CHILD functionality?
    Respond with EXACTLY "VALID" or "INVALID" followed by a brief explanation.
    """


def get_merge_prompt(group: list[FunctionalityNode]) -> str:
    """Generate prompt to determine if a group of nodes should be merged."""
    return f"""
    Analyze the following functionality nodes. Your primary goal is to determine if they represent the **same core chatbot function/action from the user's perspective**, even if there are minor variations in naming, description, or specific context mentioned (like item size or type).

    Functionality Nodes to Evaluate for Merging:
    {json.dumps([{"name": n.name, "description": n.description, "parameters": [p.to_dict() for p in n.parameters], "outputs": [o.to_dict() for o in n.outputs]} for n in group], indent=2)}

    **CRITERIA FOR MERGING (All must generally apply):**
    1.  **Functionally Equivalent Core Task:** ... The user goal addressed is identical.
    *   Example: `prompt_for_item_attributes`, `ask_for_item_customizations`, `reprompt_for_item_details` likely all serve the core task of "chatbot solicits item attribute choices from user."
    *   **Consider merging if nodes perform the same core action but differ only in specific contextual values that are, or could be, handled by parameters or are part of the conversational lead-up.** For example, `prompt_for_attributes_for_type_A_item` and `prompt_for_attributes_for_type_B_item` might merge to `prompt_for_item_attributes` if the attribute solicitation process is the same.
    2.  **Interchangeable Outcome:** From the user's perspective, successfully interacting with any of these nodes leads to the same essential outcome or state progression in the overall task.
    3.  **Similar Parameters/Outputs (Core Elements):** ... Minor differences in contextual parameters (e.g., one prompt refers to a 'large' item implicitly because 'large' was selected in a *previous* step, while another prompt is general) are acceptable if the *primary newly solicited input* (e.g., toppings, color) and the process of soliciting it are the same. The merged node would represent the general action.

    **DO NOT MERGE IF (KEEP SEPARATE if any of these are strong distinctions):**
    -   **Fundamentally Different Actions:** E.g., `list_available_attributes` (informational) vs. `prompt_for_attribute_selection` (input solicitation).
    -   **Distinct Major Workflow Paths:** One handles a "standard_item_order" and another a "custom_item_configuration" if these lead to significantly different subsequent steps or data requirements.
    -   **Different Core Objects/Topics:** E.g., `prompt_for_main_item_details` vs. `prompt_for_accessory_choice`.
    -   **Significantly Different Scope/Granularity leading to different outcomes:** E.g., `prompt_for_entire_configuration_at_once` vs. a step-by-step `prompt_for_base_model` then `prompt_for_component_selection`.

    **Your Default should be to KEEP SEPARATE unless the criteria for merging are clearly and strongly met, indicating they are just conversational variations of the same core function.**

    **EXAMPLES OF POTENTIAL MERGES (Focus on Core Function):**
    -   Group: [`prompt_for_options_for_large_item`, `ask_for_item_options`, `reprompt_user_for_option_choices`]
        Result: MERGE (if core function is "solicit option choices for an item")
        Canonical Name: `prompt_for_item_option_selection`
        Canonical Description: "Prompts the user to select or confirm desired options for their item."
    -   Group: [`show_product_menu`, `list_available_products`]
        Result: MERGE
        Canonical Name: `present_product_options`
        Canonical Description: "Presents the available product types or items to the user."
    -   Group: [`prompt_for_customizations_for_large_item`, `ask_for_customizations_for_medium_item`, `get_item_customizations`]
        Result: MERGE (if the core function is "solicit item customizations" and item size was determined previously or isn't a differentiator in *how* customizations are solicited)
        Canonical Name: `prompt_for_item_customization_options`
        Canonical Description: "Prompts the user to select or specify desired customizations for their selected item."

    **Based on these criteria, should the provided group of functionalities be merged?**

    If YES, they represent variations of the SAME core functionality:
    Respond with exactly:
    MERGE
    name: [Suggest a SINGLE, precise, canonical snake_case name for the merged function that best represents the core shared action]
    description: [Suggest a SINGLE, clear, consolidated description for the merged function that reflects the core shared purpose]

    If NO, they represent distinct functionalities that should remain separate:
    Respond with exactly:
    KEEP SEPARATE
    reason: [Briefly explain the key functional distinction(s) based on the criteria why they should NOT be merged (e.g., "Different core actions: one informs, one solicits input", "Handles distinct workflow paths with different subsequent steps")]
    """


def get_consolidate_outputs_prompt(output_details: list[dict[str, str]]) -> str:
    """Generates a prompt to consolidate a list of output specifications for a merged functionality node.

    Each dict in output_details should have "id", "category_name", "description".
    """
    output_list_str = json.dumps(output_details, indent=2)

    return f"""
You are an AI assistant tasked with consolidating a list of 'output' specifications that come from multiple similar chatbot functionalities that are being merged into one.
The goal is to produce a minimal, de-duplicated list of canonical outputs for the single merged functionality.

Here are the output specifications collected from the functionalities being merged:
{output_list_str}

**Instructions for Consolidation:**

1.  **Identify Semantic Equivalence:** Review the 'category_name' and 'description' for each output. Determine which outputs represent the exact same piece or type of information, even if their category names or descriptions are slightly different.
    *   For example, `information_topics`, `available_information_types`, and `available_topics` might all refer to the same list of what the chatbot can talk about.
    *   Similarly, `contact_email` and `support_email_address` likely refer to the same data point.
2.  **Group Equivalent Outputs:** Group the input outputs based on this semantic equivalence.
3.  **Create Canonical Representation:** For each group of equivalent outputs:
    *   Suggest a single, clear, canonical `canonical_category` name (snake_case).
    *   Write a single, comprehensive `canonical_description` that best represents the information provided by that group. If descriptions differ, try to synthesize them or pick the most complete one.
    *   List the `original_ids` of all input outputs that belong to this canonical group.
4.  **Handle Truly Distinct Outputs:** If some input outputs are genuinely distinct and not equivalent to any others, they should form their own group (possibly of size one) with their own canonical representation.

**Output Format (Strictly JSON list of objects):**
Return a JSON list where each object represents one consolidated, canonical output.

Example:
Input:
[
  {{ "id": "OUT1", "category_name": "info_topics", "description": "Topics A, B, C" }},
  {{ "id": "OUT2", "category_name": "available_topics", "description": "Can discuss A, B, C, and D" }},
  {{ "id": "OUT3", "category_name": "contact_phone_number", "description": "Main support line is 555-1234" }}
]

Expected JSON Output:
[
  {{
    "canonical_category": "knowledge_topics",
    "canonical_description": "A list of topics the chatbot can provide information about, such as A, B, C, and D.",
    "original_ids": ["OUT1", "OUT2"]
  }},
  {{
    "canonical_category": "support_phone",
    "canonical_description": "The main phone number for contacting support (e.g., 555-1234).",
    "original_ids": ["OUT3"]
  }}
]

Generate ONLY the JSON list of consolidated outputs.
"""


def get_consolidate_parameters_prompt(parameter_details: list[dict[str, Any]]) -> str:
    """Generates a prompt to consolidate a list of parameter specifications for a merged functionality node.

    Each dict in parameter_details should have "id", "name", "description", "options".
    """
    param_list_str = json.dumps(parameter_details, indent=2)

    return f"""
You are an AI assistant tasked with consolidating a list of 'parameter' specifications that come from multiple similar chatbot functionalities that are being merged into one.
The goal is to produce a minimal, de-duplicated list of canonical parameters for the single merged functionality. These parameters represent inputs the chatbot solicits.

Here are the parameter specifications collected from the functionalities being merged:
{param_list_str}

**Instructions for Consolidation:**

1.  **Identify Semantic Equivalence:** Review the 'name', 'description', and 'options' for each parameter. Determine which parameters represent the exact same piece of information the chatbot needs from the user, even if their names or descriptions are slightly different.
    *   For example, `item_id`, `product_code`, and `selected_item_identifier` might all refer to the same required input.
    *   Similarly, `size_choice` and `selected_size` could be the same.
2.  **Group Equivalent Parameters:** Group the input parameters based on this semantic equivalence.
3.  **Create Canonical Representation:** For each group of equivalent parameters:
    *   Suggest a single, clear, canonical `canonical_name` (snake_case). This should be the most representative and understandable name for the input.
    *   Write a single, comprehensive `canonical_description` that best describes what information this parameter solicits. If descriptions differ, synthesize them or pick the most complete one.
    *   Combine all unique `options` from the parameters in the group. If a parameter has no predefined options, its 'options' list will be empty. The canonical parameter should also have an empty list if none of its constituents had options.
    *   List the `original_ids` of all input parameters that belong to this canonical group.
4.  **Handle Truly Distinct Parameters:** If some input parameters are genuinely distinct and not equivalent to any others, they should form their own group (possibly of size one) with their own canonical representation.

**Output Format (Strictly JSON list of objects):**
Return a JSON list where each object represents one consolidated, canonical parameter.

Example:
Input:
[
  {{ "id": "PARAM1", "name": "item_size", "description": "The size of the item", "options": ["S", "M", "L"] }},
  {{ "id": "PARAM2", "name": "selected_size", "description": "User's chosen size", "options": ["Small", "Medium", "Large"] }},
  {{ "id": "PARAM3", "name": "delivery_zip_code", "description": "ZIP for delivery", "options": [] }}
]

Expected JSON Output:
[
  {{
    "canonical_name": "item_size",
    "canonical_description": "The desired size of the item (e.g., Small, Medium, Large).",
    "options": ["S", "M", "L", "Small", "Medium", "Large"], // Combined and unique
    "original_ids": ["PARAM1", "PARAM2"]
  }},
  {{
    "canonical_name": "delivery_zip_code",
    "canonical_description": "The postal ZIP code for the delivery address.",
    "options": [],
    "original_ids": ["PARAM3"]
  }}
]

Generate ONLY the JSON list of consolidated parameters.
"""
