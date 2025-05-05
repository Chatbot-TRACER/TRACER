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
    Analyze the following functionality nodes extracted from conversations with a potentially informational chatbot. Determine if they represent the **same core informational topic or achieve the same overall user goal**, even if the specific interaction steps (like providing options vs. displaying results vs. explaining) differ slightly.

    Functionality Nodes:
    {json.dumps([{"name": n.name, "description": n.description} for n in group], indent=2)}

    Consider the *purpose* and the *information conveyed*. For example, different ways of providing contact details (`display_contact_info`, `explain_contact_methods`, `repeat_contact_details`) should likely be merged into a single `provide_contact_info` node. However, `provide_contact_info` and `explain_ticketing_process` are distinct topics.

    If they represent the SAME core topic/goal, respond with exactly:
    MERGE
    name: [Suggest a concise, representative snake_case name for the core topic/goal, e.g., `provide_contact_info`]
    description: [Suggest a clear, consolidated description covering the core topic and potentially mentioning the different ways it was presented]

    If they are distinct topics or goals, respond with exactly:
    KEEP SEPARATE
    reason: [Briefly explain why they cover different topics/goals]
    """
