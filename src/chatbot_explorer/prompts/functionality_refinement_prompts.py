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
    Evaluate if the second functionality should be considered a sub-functionality of the first functionality.
    Use balanced judgment - we want to create a meaningful hierarchy without being overly strict.

    PARENT FUNCTIONALITY:
    Name: {parent_node.name}
    Description: {parent_node.description}

    POTENTIAL SUB-FUNCTIONALITY:
    Name: {child_node.name}
    Description: {child_node.description}

    A functionality should be considered a sub-functionality if it meets AT LEAST ONE of these criteria:
    1. It represents a more specific version or specialized case of the parent functionality
    2. It's normally used as part of completing the parent functionality
    3. It extends or enhances the parent functionality in a natural way
    4. It depends on the parent functionality conceptually or in workflow

    EXAMPLE VALID RELATIONSHIPS:
    - Parent: "search_products" - Child: "filter_search_results"
    - Parent: "schedule_appointment" - Child: "confirm_appointment_availability"
    - Parent: "estimate_price" - Child: "calculate_detailed_quote"
    - Parent: "manage_account" - Child: "update_profile_information"

    EXAMPLE INVALID RELATIONSHIPS:
    - Parent: "login" - Child: "view_product_catalog" (unrelated functions)
    - Parent: "check_weather" - Child: "translate_text" (completely different domains)

    Consider domain-specific logic and real-world workflows when making your determination.
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
