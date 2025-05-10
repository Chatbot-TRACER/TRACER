"""Module to check for duplicate functionalities, merge them and validate relationships between them."""

import json
import re

from langchain_core.language_models import BaseLanguageModel

from chatbot_explorer.prompts.functionality_refinement_prompts import (
    get_consolidate_outputs_prompt,
    get_consolidate_parameters_prompt,
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
    logger.debug("Checking if '%s' is a duplicate against %d existing nodes", node_to_check.name, len(existing_nodes))

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
        logger.debug("LLM duplicate check response: %s", result_content[:100])

        # First try exact format "DUPLICATE_OF: nodename"
        match = re.search(r"DUPLICATE_OF:\s*([\w_]+)", result_content)
        if not match:
            # Try more flexible pattern to catch more variations
            match = re.search(r"DUPLICATE.*?[\'\"]*([A-Za-z0-9_]+)[\'\"]*", result_content)

        if match:
            existing_node_name = match.group(1)
            logger.debug(
                "LLM identified potential duplicate: '%s' might match '%s'", node_to_check.name, existing_node_name
            )

            # Find the actual existing node object by this name
            for existing in existing_nodes:
                # Clean up the name for comparison if there's a newline or description
                clean_existing_name = existing.name
                if "\n" in clean_existing_name or "description:" in clean_existing_name.lower():
                    clean_existing_name = clean_existing_name.split("\n")[0].split("description:")[0].strip()

                if clean_existing_name.upper() == existing_node_name:
                    logger.debug(
                        "LLM identified semantic duplicate: '%s' matches existing '%s'",
                        node_to_check.name,
                        existing.name,
                    )
                    return True, existing  # Return the existing node object

            # If no exact match, try case-insensitive match
            for existing in existing_nodes:
                # Clean up the name for comparison if there's a newline or description
                clean_existing_name = existing.name
                if "\n" in clean_existing_name or "description:" in clean_existing_name.lower():
                    clean_existing_name = clean_existing_name.split("\n")[0].split("description:")[0].strip()

                if clean_existing_name.upper() == existing_node_name.upper():
                    logger.debug(
                        "LLM identified semantic duplicate (case-insensitive): '%s' matches existing '%s'",
                        node_to_check.name,
                        existing.name,
                    )
                    return True, existing

            # Try to find a fuzzy match based on string
            best_match = None
            for existing in existing_nodes:
                # Clean up the name for comparison
                clean_existing_name = existing.name
                if "\n" in clean_existing_name or "description:" in clean_existing_name.lower():
                    clean_existing_name = clean_existing_name.split("\n")[0].split("description:")[0].strip()

                # Check if the identified name is a substring of an existing node or vice versa
                existing_name_lower = clean_existing_name.lower()
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
        # Parse name and description from the MERGE response more robustly
        best_name = None
        best_desc = None
        lines = content.splitlines()
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith("name:"):
                if best_name is None:  # Take the first occurrence
                    best_name = line.split(":", 1)[1].strip()
            elif line_lower.startswith("description:"):
                if best_desc is None:  # Take the first occurrence
                    best_desc = line.split(":", 1)[1].strip()

        if best_name and best_desc:
            logger.debug("Parsed merged node - Name: '%s', Description: '%s'", best_name, best_desc)

            merged_node = FunctionalityNode(name=best_name, description=best_desc, parameters=[], outputs=[])

            # --- Parameter Merging ---
            all_params_from_group = []
            for node_idx, node in enumerate(group):
                for param_idx, param in enumerate(node.parameters):
                    if param and param.name:
                        all_params_from_group.append(
                            {
                                "id": f"NODE{node_idx}_PARAM{param_idx}",
                                "name": param.name,
                                "description": param.description or "",
                                "options": param.options or [],
                            }
                        )

            if not all_params_from_group:
                merged_node.parameters = []
                logger.debug("No parameters to merge for node '%s'", best_name)
            elif len(all_params_from_group) == 1 and all_params_from_group[0].get("id"):
                original_node_idx, original_param_idx = map(
                    int, all_params_from_group[0]["id"].replace("NODE", "").replace("PARAM", "").split("_")
                )
                merged_node.parameters = [group[original_node_idx].parameters[original_param_idx]]
                logger.debug(
                    "Only one parameter found across group for '%s'. Using it directly: %s",
                    best_name,
                    merged_node.parameters[0].name,
                )
            else:
                param_consolidation_prompt = get_consolidate_parameters_prompt(all_params_from_group)
                logger.debug(
                    "Consolidating parameters for merged node '%s' using LLM. Input parameters count: %d",
                    best_name,
                    len(all_params_from_group),
                )

                raw_param_consolidation_response = llm.invoke(param_consolidation_prompt).content
                logger.debug("LLM response for parameter consolidation (raw): %s", raw_param_consolidation_response)

                extracted_json_str = None
                code_block_match = re.search(
                    r"```json\s*([\s\S]*?)\s*```", raw_param_consolidation_response, re.IGNORECASE
                )
                if code_block_match:
                    extracted_json_str = code_block_match.group(1).strip()
                    logger.debug("Extracted parameter JSON from markdown code block.")
                else:
                    first_bracket = raw_param_consolidation_response.find("[")
                    last_bracket = raw_param_consolidation_response.rfind("]")
                    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                        extracted_json_str = raw_param_consolidation_response[first_bracket : last_bracket + 1]
                        logger.debug("Extracted parameter JSON by finding outermost list brackets.")

                if not extracted_json_str:  # Fallback if no structure found yet
                    extracted_json_str = raw_param_consolidation_response.strip()
                    logger.debug("Using stripped raw response as parameter JSON candidate.")

                final_params = []
                if extracted_json_str:
                    try:
                        consolidated_params_list = json.loads(extracted_json_str)
                        for consol_param in consolidated_params_list:
                            if (
                                isinstance(consol_param, dict)
                                and consol_param.get("canonical_name")
                                and "canonical_description" in consol_param
                            ):  # desc can be empty string
                                final_params.append(
                                    ParameterDefinition(
                                        name=consol_param["canonical_name"],
                                        description=consol_param["canonical_description"],
                                        options=sorted(list(set(consol_param.get("options", [])))),
                                    )
                                )
                        merged_node.parameters = final_params
                        logger.debug("Consolidated parameters for '%s': %s", best_name, [p.name for p in final_params])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            "Failed to parse JSON for parameter consolidation for node '%s'. Error: %s. Extracted string: '%s'. Using simple union by name.",
                            best_name,
                            e,
                            extracted_json_str,
                        )
                        # Fallback to old simple union by name
                        param_by_name_fallback = {}
                        for node_in_group_fb in group:
                            for param_obj_fb in node_in_group_fb.parameters:
                                if param_obj_fb.name not in param_by_name_fallback:
                                    param_by_name_fallback[param_obj_fb.name] = param_obj_fb
                                else:
                                    existing_param_obj_fb = param_by_name_fallback[param_obj_fb.name]
                                    combined_options_fb = sorted(
                                        list(set(existing_param_obj_fb.options + param_obj_fb.options))
                                    )
                                    param_by_name_fallback[param_obj_fb.name] = ParameterDefinition(
                                        name=param_obj_fb.name,
                                        description=existing_param_obj_fb.description,
                                        options=combined_options_fb,
                                    )
                        merged_node.parameters = list(param_by_name_fallback.values())
                else:  # Extracted JSON string was empty
                    logger.warning(
                        "LLM response for parameter consolidation was empty or unextractable for node '%s'. Using simple union by name.",
                        best_name,
                    )
                    # Fallback logic
                    param_by_name_fallback = {}  # Duplicated fallback, consider refactoring if too verbose
                    for node_in_group_fb in group:
                        for param_obj_fb in node_in_group_fb.parameters:
                            if param_obj_fb.name not in param_by_name_fallback:
                                param_by_name_fallback[param_obj_fb.name] = param_obj_fb
                            else:
                                existing_param_obj_fb = param_by_name_fallback[param_obj_fb.name]
                                combined_options_fb = sorted(
                                    list(set(existing_param_obj_fb.options + param_obj_fb.options))
                                )
                                param_by_name_fallback[param_obj_fb.name] = ParameterDefinition(
                                    name=param_obj_fb.name,
                                    description=existing_param_obj_fb.description,
                                    options=combined_options_fb,
                                )
                    merged_node.parameters = list(param_by_name_fallback.values())

            # --- Improved Output Merging ---

            all_outputs_from_group = []
            for node in group:
                all_outputs_from_group.extend(node.outputs)

            if not all_outputs_from_group:
                merged_node.outputs = []
            elif len(all_outputs_from_group) == 1:  # Only one output in the whole group, just use it
                merged_node.outputs = [all_outputs_from_group[0]]
            else:
                # If there are multiple outputs across the nodes being merged, try to consolidate them.
                # This requires an LLM call to group semantically similar output categories.
                output_details_for_llm = []
                for i, out_opt in enumerate(all_outputs_from_group):
                    if out_opt and out_opt.category and out_opt.description:  # Ensure valid OutputOptions object
                        output_details_for_llm.append(
                            {"id": f"OUT{i}", "category_name": out_opt.category, "description": out_opt.description}
                        )

                if output_details_for_llm:
                    consolidation_prompt = get_consolidate_outputs_prompt(output_details_for_llm)
                    logger.debug(
                        "Consolidating outputs for merged node '%s' using LLM. Input outputs: %s",
                        best_name,
                        output_details_for_llm,
                    )
                    raw_consolidation_response = llm.invoke(consolidation_prompt).content
                    logger.debug("LLM response for output consolidation (raw): %s", raw_consolidation_response)

                    extracted_json_str = None
                    # 1. Try to find JSON within markdown code blocks
                    code_block_match = re.search(
                        r"```json\s*([\s\S]*?)\s*```", raw_consolidation_response, re.IGNORECASE
                    )
                    if code_block_match:
                        extracted_json_str = code_block_match.group(1).strip()
                        logger.debug("Extracted output JSON from markdown code block.")
                    else:
                        # 2. If not in a code block, try to find the outermost list structure `[...]`
                        first_bracket = raw_consolidation_response.find("[")
                        last_bracket = raw_consolidation_response.rfind("]")
                        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                            extracted_json_str = raw_consolidation_response[first_bracket : last_bracket + 1]
                            logger.debug("Extracted output JSON by finding outermost list brackets.")
                        else:
                            # 3. Fallback: strip the raw response
                            extracted_json_str = raw_consolidation_response.strip()
                            logger.debug("Using stripped raw response as JSON candidate.")

                    if not extracted_json_str:  # Handles empty or unextractable content
                        logger.warning(
                            "Could not extract a valid JSON string from LLM response for output consolidation. Raw response: %s",
                            raw_consolidation_response,
                        )
                        # Fallback logic will be triggered by JSONDecodeError or if extracted_json_str is empty/None

                    try:
                        consolidated_outputs_list = json.loads(extracted_json_str)  # type: ignore[arg-type]
                        final_outputs = []
                        for consol_out in consolidated_outputs_list:
                            if (
                                isinstance(consol_out, dict)
                                and consol_out.get("canonical_category")
                                and consol_out.get("canonical_description")
                            ):
                                final_outputs.append(
                                    OutputOptions(
                                        category=consol_out["canonical_category"],
                                        description=consol_out["canonical_description"],
                                    )
                                )
                        merged_node.outputs = final_outputs
                        logger.debug(
                            "Consolidated outputs for '%s': %s", best_name, [o.category for o in final_outputs]
                        )
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            "Failed to parse JSON for output consolidation for node '%s'. Error: %s. Extracted string: '%s'. Using simple union of unique categories.",
                            best_name,
                            e,
                            extracted_json_str,
                        )
                        # Fallback to simple unique category union if LLM fails
                        fallback_output_by_category = {}
                        for out_opt in all_outputs_from_group:
                            if out_opt and out_opt.category and out_opt.category not in fallback_output_by_category:
                                fallback_output_by_category[out_opt.category] = out_opt
                        merged_node.outputs = list(fallback_output_by_category.values())
                else:  # No valid outputs to process
                    merged_node.outputs = []

            # Merge children too
            for node in group:
                for child in node.children:
                    child.parent = merged_node
                    merged_node.add_child(child)

            logger.debug("Merged %d functionalities into '%s'", len(group), best_name)
            return [merged_node]

        logger.warning(
            "LLM suggested MERGE, but could not parse name/description for group: %s. Content: '%s'. Keeping first node '%s'",
            [n.name for n in group],
            content[:200],
            group[0].name,
        )
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
