import json
from typing import List
import re
from ...functionality_node import FunctionalityNode


def is_duplicate_functionality(
    node: FunctionalityNode, existing_nodes: List[FunctionalityNode], llm=None
) -> bool:
    """
    Check if this node is basically the same as one we already have.

    Args:
        node (FunctionalityNode): The new node to check.
        existing_nodes (list): List of nodes already found.
        llm (optional): The language model instance for semantic check. Defaults to None.

    Returns:
        bool: True if it seems like a duplicate, False otherwise.
    """
    # Simple checks first
    for existing in existing_nodes:
        # Exact name or description match
        if (
            existing.name.lower() == node.name.lower()
            or existing.description.lower() == node.description.lower()
        ):
            return True

    # Use LLM for smarter check if available
    if llm and existing_nodes:
        # Limit checks to save API calls
        nodes_to_check = existing_nodes[:5]

        # Format existing nodes for prompt
        existing_descriptions = [
            f"Name: {n.name}, Description: {n.description}" for n in nodes_to_check
        ]

        # Prompt for LLM
        duplicate_check_prompt = f"""
        Determine if the new functionality is semantically equivalent to any existing functionality.

        NEW FUNCTIONALITY:
        Name: {node.name}
        Description: {node.description}

        EXISTING FUNCTIONALITIES:
        {json.dumps(existing_descriptions, indent=2)}

        A functionality is a duplicate if it represents the SAME ACTION/CAPABILITY, even if described differently.

        Respond with ONLY "DUPLICATE" or "UNIQUE" followed by a brief explanation.
        """

        response = llm.invoke(duplicate_check_prompt)
        result = response.content.strip().upper()

        if "DUPLICATE" in result:
            return True

    return False


def validate_parent_child_relationship(parent_node, child_node, llm) -> bool:
    """
    Check if the child node makes sense as a sub-step of the parent node.

    Args:
        parent_node (FunctionalityNode): The potential parent node.
        child_node (FunctionalityNode): The potential child node.
        llm: The language model instance.

    Returns:
        bool: True if the relationship seems valid, False otherwise.
    """
    if not parent_node:
        return True  # Root nodes are always valid

    # Prompt for LLM validation
    validation_prompt = f"""
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

    # Get LLM response
    validation_response = llm.invoke(validation_prompt)
    result = validation_response.content.strip().upper()

    is_valid = result.startswith("VALID")

    if is_valid:
        print(
            f"  ✓ Valid relationship: '{child_node.name}' is a sub-functionality of '{parent_node.name}'"
        )
    else:
        print(
            f"  ✗ Invalid relationship: '{child_node.name}' is not related to '{parent_node.name}'"
        )

    return is_valid


def merge_similar_functionalities(
    nodes: List[FunctionalityNode], llm
) -> List[FunctionalityNode]:
    """
    Use LLM to find and merge similar nodes. Returns a new list.

    Args:
        nodes (list): List of FunctionalityNode objects to check.
        llm: The language model instance.

    Returns:
        list: A new list of FunctionalityNode objects with similar ones merged.
    """
    if not nodes or len(nodes) < 2:
        return nodes  # Nothing to merge

    result = []
    name_groups = {}

    # Group nodes by normalized name first
    for node in nodes:
        normalized_name = node.name.lower().replace("_", " ")
        if normalized_name not in name_groups:
            name_groups[normalized_name] = []
        name_groups[normalized_name].append(node)

    for name, group in name_groups.items():
        if len(group) == 1:
            # Only one node, keep it
            result.append(group[0])
            continue

        # Ask LLM if this group should be merged
        merge_prompt = f"""
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

        merge_response = llm.invoke(merge_prompt)
        content = merge_response.content.strip()

        if content.upper().startswith("MERGE"):
            # Try to parse suggested name and description
            name_match = re.search(r"name:\s*(.+)", content)
            desc_match = re.search(r"description:\s*(.+)", content)
            if name_match and desc_match:
                best_name = name_match.group(1).strip()
                best_desc = desc_match.group(1).strip()

                # Combine parameters and children from the group
                all_params = []
                merged_node = FunctionalityNode(
                    name=best_name, description=best_desc, parameters=all_params
                )

                for node in group:
                    # Merge parameters (avoid duplicates)
                    for param in node.parameters:
                        if not any(
                            p.get("name") == param.get("name") for p in all_params
                        ):
                            all_params.append(param)
                    # Add all children
                    for child in node.children:
                        merged_node.add_child(child)

                print(f"  Merged {len(group)} functionalities into '{best_name}'")
                result.append(merged_node)
            else:
                # Fallback if parsing fails: keep the first node
                print(
                    f"  WARN: Could not parse merge suggestion for group '{name}'. Keeping first node."
                )
                result.append(group[0])
        else:
            # Keep nodes separate if LLM says so
            result.extend(group)

    return result
