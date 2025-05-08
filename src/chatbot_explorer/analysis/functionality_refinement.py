"""Module to check for duplicate functionalities, merge them and validate relationships between them."""

import re

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.functionality_refinement_prompts import (
    get_duplicate_check_prompt,
    get_merge_prompt,
    get_relationship_validation_prompt,
)
from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode, OutputOptions, ParameterDefinition
from chatbot_explorer.utils.logging_utils import get_logger

logger = get_logger()


def is_duplicate_functionality(
    node_to_check: FunctionalityNode, existing_nodes: list[FunctionalityNode], llm: BaseLanguageModel | None = None
) -> tuple[bool, FunctionalityNode | None]:
    """Checks if a given node is semantically equivalent to any node in a list of existing nodes.

    Uses simple string comparison first, if no duplicates are found, it uses an LLM to check for semantic equivalence.

    Args:
        node_to_check: The node to check for duplicates.
        existing_nodes: A list of nodes already discovered.
        llm: The language model instance, if available for semantic checking.

    Returns:
        True if the node is considered a duplicate, False otherwise.
        If True, also returns the existing node it matches with.
    """
    # Simple checks first
    for existing in existing_nodes:
        if (
            existing.name.lower() == node_to_check.name.lower()
            or existing.description.lower() == node_to_check.description.lower()
        ):
            logger.debug("Found exact match duplicate: '%s' matches existing '%s'", node_to_check.name, existing.name)
            return True, existing

    # Use LLM for smarter check if available and if there are nodes to compare against
    if llm and existing_nodes:
        nodes_to_check = existing_nodes
        existing_descriptions = [f"Name: {n.name}, Description: {n.description}" for n in nodes_to_check]

        duplicate_check_prompt = get_duplicate_check_prompt(
            node=node_to_check,
            existing_descriptions=existing_descriptions,
        )

        logger.debug("Checking if '%s' is semantically equivalent to any existing node", node_to_check.name)
        response = llm.invoke(duplicate_check_prompt)
        result_content = response.content.strip().upper()

        match = re.search(r"DUPLICATE_OF:\s*([\w_]+)", result_content)
        if match:
            existing_node_name = match.group(1)
            # Find the actual existing node object by this name
            exact_match = None
            for existing in existing_nodes:
                if existing.name.upper() == existing_node_name:
                    logger.debug(
                        "LLM identified semantic duplicate: '%s' matches existing '%s'",
                        node_to_check.name,
                        existing.name,
                    )
                    return True, existing  # Return the existing node object

            # If no exact match, try case-insensitive match
            for existing in existing_nodes:
                if existing.name.upper() == existing_node_name.upper():
                    logger.debug(
                        "LLM identified semantic duplicate (case-insensitive): '%s' matches existing '%s'",
                        node_to_check.name,
                        existing.name,
                    )
                    return True, existing

            # Try to find a fuzzy match based on string
            best_match = None
            for existing in existing_nodes:
                # Check if the identified name is a substring of an existing node or vice versa
                existing_name_lower = existing.name.lower()
                identified_name_lower = existing_node_name.lower()

                if existing_name_lower in identified_name_lower or identified_name_lower in existing_name_lower:
                    best_match = existing
                    logger.debug(
                        "Found fuzzy match: LLM identified '%s', matched with existing '%s'",
                        existing_node_name,
                        existing.name,
                    )
                    return True, best_match

            # If still no match found, log warning
            logger.warning(
                "LLM said DUPLICATE_OF '%s' but node not found in existing_nodes list.", existing_node_name
            )  # Should not happen if prompt is good

            return False, None  # Fallback, treat as unique if named node not found

        if "DUPLICATE" in result_content:  # Generic duplicate if specific name isn't parsed
            logger.warning(
                "LLM said DUPLICATE but couldn't parse specific match name for '%s'. Trying to find a close match.",
                node_to_check.name,
            )
            # Return the most similar node by name length ratio as a heuristic
            if existing_nodes:
                logger.info(
                    "Using first existing node '%s' as a proxy match for '%s'",
                    existing_nodes[0].name,
                    node_to_check.name,
                )
                return True, existing_nodes[0]
            return False, None

    return False, None


def validate_parent_child_relationship(
    parent_node: FunctionalityNode, child_node: FunctionalityNode, llm: BaseLanguageModel
) -> bool:
    """Uses an LLM to determine if a child node logically follows or is a sub-step of a parent node.

    Args:
        parent_node: The potential parent node.
        child_node: The potential child node.
        llm: The language model instance.

    Returns:
        True if the relationship is deemed valid by the LLM, False otherwise.
    """
    if not parent_node:
        return True

    validation_prompt = get_relationship_validation_prompt(
        parent_node=parent_node,
        child_node=child_node,
    )

    validation_response = llm.invoke(validation_prompt)
    result = validation_response.content.strip().upper()

    is_valid = result.startswith("VALID")

    # Log the validation result at debug level
    if is_valid:
        logger.debug("✓ Valid relationship: '%s' is a sub-functionality of '%s'", child_node.name, parent_node.name)
    else:
        logger.debug("✗ Invalid relationship: '%s' is not related to '%s'", child_node.name, parent_node.name)

    return is_valid


def _process_node_group_for_merge(group: list[FunctionalityNode], llm: BaseLanguageModel) -> list[FunctionalityNode]:
    """Processes a group of nodes (assumed to have similar names) to potentially merge them into one.

    Uses an LLM to suggest whether to merge and, if so, determines the best name and description
    for the merged node. It combines parameters and children from the original nodes.

    Args:
        group: A list of FunctionalityNode objects with similar names.
        llm: The language model instance.

    Returns:
        A list containing either a single merged node or the original nodes if no merge occurred
        or if merging failed.
    """
    if len(group) == 1:
        return group

    # Log what we're trying to merge
    node_names = [node.name for node in group]
    logger.debug("Evaluating potential merge of %d nodes: %s", len(group), ", ".join(node_names))

    merge_prompt = get_merge_prompt(group=group)
    merge_response = llm.invoke(merge_prompt)
    content = merge_response.content.strip()
    logger.debug("Merge response content: '%s'", content)

    if content.upper().startswith("MERGE"):
        name_match = re.search(r"name:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        desc_match = re.search(r"description:\s*(.+)", content, re.IGNORECASE | re.DOTALL)

        if name_match and desc_match:
            best_name = name_match.group(1).strip()
            best_desc = desc_match.group(1).strip()

            merged_node = FunctionalityNode(name=best_name, description=best_desc, parameters=[], outputs=[])

            param_by_name = {}

            for node in group:
                for param in node.parameters:
                    param_name = param.name

                    if param_name not in param_by_name:
                        param_by_name[param_name] = param
                    else:
                        existing_param = param_by_name[param_name]

                        combined_options = list(set(existing_param.options + param.options))

                        merged_param = ParameterDefinition(
                            name=param_name, description=existing_param.description, options=combined_options
                        )
                        param_by_name[param_name] = merged_param

            output_by_category = {}

            for node in group:
                for output in node.outputs:
                    category = output.category

                    if category not in output_by_category:
                        output_by_category[category] = output
                    else:
                        existing_output = output_by_category[category]

                        description = (
                            output.description
                            if len(output.description) > len(existing_output.description)
                            else existing_output.description
                        )

                        merged_output = OutputOptions(category=category, description=description)
                        output_by_category[category] = merged_output

            # Convert merged parameters to a list
            merged_node.parameters = list(param_by_name.values())

            # Convert merged output options to a list
            merged_node.outputs = list(output_by_category.values())

            # Merge children too
            for node in group:
                for child in node.children:
                    child.parent = merged_node
                    merged_node.add_child(child)

            logger.debug("Merged %d functionalities into '%s'", len(group), best_name)
            return [merged_node]

        logger.warning("Could not parse merge suggestion for group. Keeping first node '%s'", group[0].name)
        return [group[0]]

    logger.debug("LLM suggested keeping %d nodes separate", len(group))
    return group


def merge_similar_functionalities(nodes: list[FunctionalityNode], llm: BaseLanguageModel) -> list[FunctionalityNode]:
    """Identifies and merges similar FunctionalityNode objects within a list based on name grouping and LLM validation.

    Groups nodes by a normalized version of their names, then uses an LLM via the
    `_process_node_group_for_merge` helper function to decide whether to merge nodes within each group.

    Args:
        nodes: The initial list of FunctionalityNode objects to process.
        llm: The language model instance used for merge decisions.

    Returns:
        A new list of FunctionalityNode objects where similar nodes may have been merged.
    """
    min_nodes_to_merge = 2
    if not nodes or len(nodes) < min_nodes_to_merge:
        return nodes

    logger.debug("Checking for potentially similar functionalities to merge among %d nodes", len(nodes))

    # Group nodes by similar names
    name_groups: dict[str, list[FunctionalityNode]] = {}
    for node in nodes:
        normalized_name = node.name.lower().replace("_", " ")
        if normalized_name not in name_groups:
            name_groups[normalized_name] = []
        name_groups[normalized_name].append(node)

    # Process each group that may need merging
    merged_results: list[FunctionalityNode] = []
    groups_that_need_merging = [group for group in name_groups.values() if len(group) > 1]

    if groups_that_need_merging:
        logger.debug("Found %d groups with potential duplicate functionalities", len(groups_that_need_merging))

    # Process all groups
    for group in name_groups.values():
        processed_group = _process_node_group_for_merge(group, llm)
        merged_results.extend(processed_group)

    # Log results if any merging happened
    if len(merged_results) < len(nodes):
        logger.verbose("Merged %d nodes into %d nodes after similarity analysis", len(nodes), len(merged_results))
    else:
        logger.debug("No nodes were merged. Keeping original %d nodes", len(nodes))

    return merged_results
