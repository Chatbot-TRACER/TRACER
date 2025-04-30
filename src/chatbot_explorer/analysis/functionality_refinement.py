import re

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.functionality_refinement_prompts import (
    get_duplicate_check_prompt,
    get_merge_prompt,
    get_relationship_validation_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode


def is_duplicate_functionality(
    node: FunctionalityNode, existing_nodes: list[FunctionalityNode], llm: BaseLanguageModel | None = None
) -> bool:
    """Check if this node is basically the same as one we already have."""
    # Simple checks first
    for existing in existing_nodes:
        # Exact name or description match
        if existing.name.lower() == node.name.lower() or existing.description.lower() == node.description.lower():
            return True

    # Use LLM for smarter check if available
    if llm and existing_nodes:
        # Limit checks to save API calls
        nodes_to_check = existing_nodes[:5]

        # Format existing nodes for prompt
        existing_descriptions = [f"Name: {n.name}, Description: {n.description}" for n in nodes_to_check]

        duplicate_check_prompt = get_duplicate_check_prompt(
            node=node,
            existing_descriptions=existing_descriptions,
        )

        response = llm.invoke(duplicate_check_prompt)
        result = response.content.strip().upper()

        if "DUPLICATE" in result:
            return True

    return False


def validate_parent_child_relationship(
    parent_node: FunctionalityNode, child_node: FunctionalityNode, llm: BaseLanguageModel
) -> bool:
    """Check if the child node makes sense as a sub-step of the parent node."""
    if not parent_node:
        return True  # Root nodes are always valid

    validation_prompt = get_relationship_validation_prompt(
        parent_node=parent_node,
        child_node=child_node,
    )

    # Get LLM response
    validation_response = llm.invoke(validation_prompt)
    result = validation_response.content.strip().upper()

    is_valid = result.startswith("VALID")

    if is_valid:
        print(f"  ✓ Valid relationship: '{child_node.name}' is a sub-functionality of '{parent_node.name}'")
    else:
        print(f"  ✗ Invalid relationship: '{child_node.name}' is not related to '{parent_node.name}'")

    return is_valid


def merge_similar_functionalities(nodes: list[FunctionalityNode], llm: BaseLanguageModel) -> list[FunctionalityNode]:
    """Use LLM to find and merge similar nodes. Returns a new list."""
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

        merge_prompt = get_merge_prompt(group=group)

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
                merged_node = FunctionalityNode(name=best_name, description=best_desc, parameters=all_params)

                for node in group:
                    # Merge parameters (avoid duplicates)
                    for param in node.parameters:
                        if not any(p.get("name") == param.get("name") for p in all_params):
                            all_params.append(param)
                    # Add all children
                    for child in node.children:
                        merged_node.add_child(child)

                print(f"  Merged {len(group)} functionalities into '{best_name}'")
                result.append(merged_node)
            else:
                # Fallback if parsing fails: keep the first node
                print(f"  WARN: Could not parse merge suggestion for group '{name}'. Keeping first node.")
                result.append(group[0])
        else:
            # Keep nodes separate if LLM says so
            result.extend(group)

    return result
