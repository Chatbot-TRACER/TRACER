"""Module to check for duplicate functionalities, merge them and validate relationships between them."""

import json
import re

from langchain_core.language_models import BaseLanguageModel

from tracer.prompts.functionality_refinement_prompts import (
    get_consolidate_outputs_prompt,
    get_consolidate_parameters_prompt,
    get_duplicate_check_prompt,
    get_merge_prompt,
    get_relationship_validation_prompt,
)
from tracer.schemas.functionality_node_model import FunctionalityNode, OutputOptions, ParameterDefinition
from tracer.utils.logging_utils import get_logger

logger = get_logger()


def _check_exact_duplicates(
    node_to_check: FunctionalityNode, existing_nodes: list[FunctionalityNode]
) -> tuple[bool, FunctionalityNode | None]:
    """Check for exact string matches between nodes."""
    for existing in existing_nodes:
        if (
            existing.name.lower() == node_to_check.name.lower()
            or existing.description.lower() == node_to_check.description.lower()
        ):
            logger.debug("Found exact match duplicate: '%s' matches existing '%s'", node_to_check.name, existing.name)
            return True, existing
    return False, None


def _clean_node_name(name: str) -> str:
    """Clean up node name for comparison by removing descriptions and newlines."""
    if "\n" in name or "description:" in name.lower():
        return name.split("\n")[0].split("description:")[0].strip()
    return name


def _find_exact_match(existing_nodes: list[FunctionalityNode], target_name: str) -> FunctionalityNode | None:
    """Find exact match by name (case-sensitive)."""
    for existing in existing_nodes:
        clean_name = _clean_node_name(existing.name)
        if clean_name.upper() == target_name:
            return existing
    return None


def _find_case_insensitive_match(existing_nodes: list[FunctionalityNode], target_name: str) -> FunctionalityNode | None:
    """Find case-insensitive match by name."""
    for existing in existing_nodes:
        clean_name = _clean_node_name(existing.name)
        if clean_name.upper() == target_name.upper():
            return existing
    return None


def _find_fuzzy_match(existing_nodes: list[FunctionalityNode], target_name: str) -> FunctionalityNode | None:
    """Find fuzzy match using substring matching."""
    target_lower = target_name.lower()
    for existing in existing_nodes:
        clean_name = _clean_node_name(existing.name)
        existing_lower = clean_name.lower()

        if existing_lower in target_lower or target_lower in existing_lower:
            logger.debug(
                "Found fuzzy match: LLM identified '%s', matched with existing '%s'",
                target_name,
                existing.name,
            )
            return existing
    return None


def _perform_llm_duplicate_check(
    node_to_check: FunctionalityNode, existing_nodes: list[FunctionalityNode], llm: BaseLanguageModel
) -> tuple[bool, FunctionalityNode | None]:
    """Perform LLM-based duplicate checking."""
    existing_descriptions = [f"Name: {n.name}, Description: {n.description}" for n in existing_nodes]

    duplicate_check_prompt = get_duplicate_check_prompt(
        node=node_to_check,
        existing_descriptions=existing_descriptions,
    )

    logger.debug("Checking if '%s' is semantically equivalent to any existing node", node_to_check.name)
    response = llm.invoke(duplicate_check_prompt)
    result_content = response.content.strip().upper()
    logger.debug("LLM duplicate check response: %s", result_content[:100])

    # Try to extract node name from response
    match = re.search(r"DUPLICATE_OF:\s*([\w_]+)", result_content)
    if not match:
        match = re.search(r"DUPLICATE.*?[\'\"]*([A-Za-z0-9_]+)[\'\"]*", result_content)

    if match:
        existing_node_name = match.group(1)
        logger.debug(
            "LLM identified potential duplicate: '%s' might match '%s'", node_to_check.name, existing_node_name
        )

        # Try different matching strategies
        matched_node = (
            _find_exact_match(existing_nodes, existing_node_name)
            or _find_case_insensitive_match(existing_nodes, existing_node_name)
            or _find_fuzzy_match(existing_nodes, existing_node_name)
        )

        if matched_node:
            logger.debug(
                "LLM identified semantic duplicate: '%s' matches existing '%s'",
                node_to_check.name,
                matched_node.name,
            )
            return True, matched_node

        logger.warning("LLM said DUPLICATE_OF '%s' but node not found in existing_nodes list.", existing_node_name)
        return False, None

    if "DUPLICATE" in result_content:
        logger.warning(
            "LLM said DUPLICATE but couldn't parse specific match name for '%s'. Trying to find a close match.",
            node_to_check.name,
        )
        if existing_nodes:
            logger.info(
                "Using first existing node '%s' as a proxy match for '%s'",
                existing_nodes[0].name,
                node_to_check.name,
            )
            return True, existing_nodes[0]

    return False, None


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
    logger.debug("Checking if '%s' is a duplicate against %d existing nodes", node_to_check.name, len(existing_nodes))

    # Check for exact duplicates first
    is_duplicate, duplicate_node = _check_exact_duplicates(node_to_check, existing_nodes)
    if is_duplicate:
        return True, duplicate_node

    # Use LLM for semantic checking if available
    if llm and existing_nodes:
        return _perform_llm_duplicate_check(node_to_check, existing_nodes, llm)

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


def _parse_merge_response(content: str) -> tuple[str | None, str | None]:
    """Parse name and description from LLM merge response."""
    best_name = None
    best_desc = None
    lines = content.splitlines()
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("name:") and best_name is None:
            best_name = line.split(":", 1)[1].strip()
        elif line_lower.startswith("description:") and best_desc is None:
            best_desc = line.split(":", 1)[1].strip()
    return best_name, best_desc


def _collect_parameters_from_group(group: list[FunctionalityNode]) -> list[dict]:
    """Collect all parameters from nodes in a group."""
    all_params = []
    for node_idx, node in enumerate(group):
        for param_idx, param in enumerate(node.parameters):
            if param and param.name:
                all_params.append(
                    {
                        "id": f"NODE{node_idx}_PARAM{param_idx}",
                        "name": param.name,
                        "description": param.description or "",
                        "options": param.options or [],
                    }
                )
    return all_params


def _create_single_parameter(param_data: dict, group: list[FunctionalityNode]) -> list[ParameterDefinition]:
    """Create parameter list when only one parameter exists across the group."""
    original_node_idx, original_param_idx = map(
        int, param_data["id"].replace("NODE", "").replace("PARAM", "").split("_")
    )
    return [group[original_node_idx].parameters[original_param_idx]]


def _extract_json_from_response(response: str, data_type: str) -> str | None:
    """Extract JSON string from LLM response."""
    # Try to find JSON within markdown code blocks
    code_block_match = re.search(r"```json\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
    if code_block_match:
        logger.debug("Extracted %s JSON from markdown code block.", data_type)
        return code_block_match.group(1).strip()

    # Try to find outermost list structure
    first_bracket = response.find("[")
    last_bracket = response.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        logger.debug("Extracted %s JSON by finding outermost list brackets.", data_type)
        return response[first_bracket : last_bracket + 1]

    # Fallback: use stripped response
    logger.debug("Using stripped raw response as %s JSON candidate.", data_type)
    return response.strip()


def _consolidate_parameters_with_llm(
    all_params: list[dict], llm: BaseLanguageModel, node_name: str
) -> list[ParameterDefinition]:
    """Consolidate parameters using LLM."""
    param_consolidation_prompt = get_consolidate_parameters_prompt(all_params)
    logger.debug(
        "Consolidating parameters for merged node '%s' using LLM. Input parameters count: %d",
        node_name,
        len(all_params),
    )

    raw_response = llm.invoke(param_consolidation_prompt).content
    logger.debug("LLM response for parameter consolidation (raw): %s", raw_response)

    extracted_json_str = _extract_json_from_response(raw_response, "parameter")

    if not extracted_json_str:
        logger.warning("Could not extract JSON from parameter consolidation response")
        return []

    try:
        consolidated_params_list = json.loads(extracted_json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            "Failed to parse JSON for parameter consolidation for node '%s'. Error: %s. Using fallback.", node_name, e
        )
        return []
    else:
        final_params = [
            ParameterDefinition(
                name=consol_param["canonical_name"],
                description=consol_param["canonical_description"],
                options=sorted(set(consol_param.get("options", []))),
            )
            for consol_param in consolidated_params_list
            if (
                isinstance(consol_param, dict)
                and consol_param.get("canonical_name")
                and "canonical_description" in consol_param
            )
        ]
        logger.debug("Consolidated parameters for '%s': %s", node_name, [p.name for p in final_params])
        return final_params


def _create_fallback_parameters(group: list[FunctionalityNode]) -> list[ParameterDefinition]:
    """Create parameters using simple union by name as fallback."""
    param_by_name = {}
    for node in group:
        for param in node.parameters:
            if param.name not in param_by_name:
                param_by_name[param.name] = param
            else:
                existing_param = param_by_name[param.name]
                combined_options = sorted(set(existing_param.options + param.options))
                param_by_name[param.name] = ParameterDefinition(
                    name=param.name,
                    description=existing_param.description,
                    options=combined_options,
                )
    return list(param_by_name.values())


def _merge_parameters(
    group: list[FunctionalityNode], llm: BaseLanguageModel, node_name: str
) -> list[ParameterDefinition]:
    """Merge parameters from a group of nodes."""
    all_params = _collect_parameters_from_group(group)

    if not all_params:
        logger.debug("No parameters to merge for node '%s'", node_name)
        return []

    if len(all_params) == 1:
        logger.debug("Only one parameter found across group for '%s'. Using it directly.", node_name)
        return _create_single_parameter(all_params[0], group)

    # Try LLM consolidation first
    consolidated_params = _consolidate_parameters_with_llm(all_params, llm, node_name)
    if consolidated_params:
        return consolidated_params

    # Fallback to simple union
    logger.warning("Using simple union by name for parameters in node '%s'", node_name)
    return _create_fallback_parameters(group)


def _collect_outputs_from_group(group: list[FunctionalityNode]) -> list[OutputOptions]:
    """Collect all outputs from nodes in a group."""
    all_outputs = []
    for node in group:
        all_outputs.extend(node.outputs)
    return all_outputs


def _consolidate_outputs_with_llm(
    output_details: list[dict], llm: BaseLanguageModel, node_name: str
) -> list[OutputOptions]:
    """Consolidate outputs using LLM."""
    consolidation_prompt = get_consolidate_outputs_prompt(output_details)
    logger.debug("Consolidating outputs for merged node '%s' using LLM. Input outputs: %s", node_name, output_details)

    raw_response = llm.invoke(consolidation_prompt).content
    logger.debug("LLM response for output consolidation (raw): %s", raw_response)

    extracted_json_str = _extract_json_from_response(raw_response, "output")

    if not extracted_json_str:
        logger.warning("Could not extract JSON from output consolidation response")
        return []

    try:
        consolidated_outputs_list = json.loads(extracted_json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            "Failed to parse JSON for output consolidation for node '%s'. Error: %s. Using fallback.", node_name, e
        )
        return []
    else:
        final_outputs = [
            OutputOptions(
                category=consol_out["canonical_category"],
                description=consol_out["canonical_description"],
            )
            for consol_out in consolidated_outputs_list
            if (
                isinstance(consol_out, dict)
                and consol_out.get("canonical_category")
                and consol_out.get("canonical_description")
            )
        ]
        logger.debug("Consolidated outputs for '%s': %s", node_name, [o.category for o in final_outputs])
        return final_outputs


def _create_fallback_outputs(all_outputs: list[OutputOptions]) -> list[OutputOptions]:
    """Create outputs using simple unique category union as fallback."""
    output_by_category = {}
    for output in all_outputs:
        if output and output.category and output.category not in output_by_category:
            output_by_category[output.category] = output
    return list(output_by_category.values())


def _merge_outputs(group: list[FunctionalityNode], llm: BaseLanguageModel, node_name: str) -> list[OutputOptions]:
    """Merge outputs from a group of nodes."""
    all_outputs = _collect_outputs_from_group(group)

    if not all_outputs:
        return []

    if len(all_outputs) == 1:
        return [all_outputs[0]]

    # Prepare output details for LLM
    output_details = [
        {"id": f"OUT{i}", "category_name": output.category, "description": output.description}
        for i, output in enumerate(all_outputs)
        if output and output.category and output.description
    ]

    if not output_details:
        return []

    # Try LLM consolidation first
    consolidated_outputs = _consolidate_outputs_with_llm(output_details, llm, node_name)
    if consolidated_outputs:
        return consolidated_outputs

    # Fallback to simple union
    logger.warning("Using simple union by category for outputs in node '%s'", node_name)
    return _create_fallback_outputs(all_outputs)


def _merge_children(group: list[FunctionalityNode], merged_node: FunctionalityNode) -> None:
    """Merge children from all nodes in the group to the merged node."""
    for node in group:
        for child in node.children:
            child.parent = merged_node
            merged_node.add_child(child)


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

    if not content.upper().startswith("MERGE"):
        logger.debug("LLM suggested keeping %d nodes separate", len(group))
        return group

    # Parse name and description from the MERGE response
    best_name, best_desc = _parse_merge_response(content)

    if not (best_name and best_desc):
        logger.warning(
            "LLM suggested MERGE, but could not parse name/description for group: %s. Content: '%s'. Keeping first node '%s'",
            [n.name for n in group],
            content[:200],
            group[0].name,
        )
        return [group[0]]

    logger.debug("Parsed merged node - Name: '%s', Description: '%s'", best_name, best_desc)

    # Create merged node
    merged_node = FunctionalityNode(name=best_name, description=best_desc, parameters=[], outputs=[])

    # Merge parameters, outputs, and children
    merged_node.parameters = _merge_parameters(group, llm, best_name)
    merged_node.outputs = _merge_outputs(group, llm, best_name)
    _merge_children(group, merged_node)

    logger.debug("Merged %d functionalities into '%s'", len(group), best_name)
    return [merged_node]


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
