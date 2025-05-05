"""Prompts for refining functionality nodes based on existing descriptions."""

import json

from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode


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

    Respond with ONLY "DUPLICATE" or "UNIQUE" followed by a brief explanation.
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
    Analyze the following functionality nodes extracted from conversations with a chatbot.
    Determine if they represent the **EXACT SAME specific function or action from the chatbot's perspective**, or if they are distinct functionalities that should remain separate. Your default should be to KEEP SEPARATE unless criteria for merging are definitively met.

    Functionality Nodes:
    {json.dumps([{"name": n.name, "description": n.description} for n in group], indent=2)}

    **CRITICAL CRITERIA FOR MERGING (Must meet BOTH):**
    1.  **Identical Goal & Action:** The nodes describe the *precise same user goal* being addressed by the *precise same chatbot action*. Minor wording differences in description are acceptable ONLY IF the underlying function is identical.
    2.  **No Functional Distinction:** There is NO difference in the *type* of task, the *level of detail* handled, the *specific workflow path* initiated, or the *object* being acted upon. They must be functionally interchangeable.

    **DO NOT MERGE IF ANY OF THESE APPLY (KEEP SEPARATE):**
    -   **Different Variants/Paths:** One node handles a standard option/path, while another handles a custom/configurable/alternative option or path (e.g., `show_standard_packages` vs. `start_custom_config`).
    -   **Different Workflow Steps:** The nodes represent distinct sequential steps in a larger process (e.g., `select_item` vs. `add_to_cart` vs. `proceed_to_checkout`).
    -   **Different Objects/Topics:** The nodes perform similar actions but on fundamentally different subjects or data types (e.g., `get_product_details` vs. `get_shipping_info`; `order_main_item` vs. `order_side_item`).
    -   **General vs. Specific:** One node is a general category or entry point, while the other is a specific sub-task, outcome, or refinement (e.g., `manage_account` vs. `update_email_address`; `search_items` vs. `apply_filter`).
    -   **Information vs. Action:** One node primarily provides information, while the other performs a state-changing or transactional action, even if related (e.g., `list_available_options` vs. `select_option`).

    **EXAMPLES:**
    -   **MERGE Candidates:**
        - `display_contact_details` and `show_contact_information` (Likely identical function)
        - `get_order_status_update` and `check_order_progress` (Likely identical function)
    -   **KEEP SEPARATE Examples:**
        - `provide_standard_options` vs. `initiate_custom_build` (Different Variants/Paths)
        - `view_cart` vs. `proceed_to_payment` (Different Workflow Steps)
        - `order_pizza` vs. `order_drink` (Different Objects/Topics - **assuming these were extracted separately**)
        - `explain_return_policy` vs. `initiate_return_request` (Information vs. Action)
        - `search_products` vs. `filter_search_results` (General vs. Specific / Different Workflow Steps)

    If they represent the **EXACT SAME** functionality based *only* on wording differences and meet the strict merge criteria, respond with exactly:
    MERGE
    name: [Suggest a precise, representative snake_case name for the identical function]
    description: [Suggest a clear, consolidated description reflecting the identical core function]

    If they are distinct functionalities according to the criteria above (or if unsure), respond with exactly:
    KEEP SEPARATE
    reason: [Briefly explain the key functional distinction based on the criteria (e.g., "Different Variants: Standard vs Custom", "Different Workflow Steps: View vs Pay")]
    """
