"""Functions for building the workflow graph structure using LLM analysis."""

import json
import re
from typing import Any, Dict, List


def build_workflow_structure(
    flat_functionality_dicts: List[Dict[str, Any]],
    conversation_history: List[Any],
    chatbot_type: str,
    llm,
) -> List[Dict[str, Any]]:
    """Build a hierarchical structure of chatbot functionalities.

    Args:
        flat_functionality_dicts: List of functionality dictionaries
        conversation_history: List of conversation sessions
        chatbot_type: Classification of the bot ("transactional" or "informational")
        llm: The language model instance

    Returns:
        List[Dict[str, Any]]: Structured hierarchy with parent-child relationships
    """
    print("\n--- Building Workflow Structure ---")

    if not flat_functionality_dicts:
        print("   Skipping structure building: No functionalities found.")
        return []

    # Format functionality list as a string for the prompt
    func_list_str = "\n".join(
        [
            f"- Name: {f.get('name', 'N/A')}\n  Description: {f.get('description', 'N/A')}\n  Parameters: {', '.join(p.get('name', '?') for p in f.get('parameters', [])) or 'None'}"
            for f in flat_functionality_dicts
        ]
    )

    # Get conversation snippets for context
    snippets = []
    total_snippet_length = 0
    max_total_snippet_length = 7000  # Larger context for structure analysis

    if isinstance(conversation_history, list):
        for i, session_history in enumerate(conversation_history):
            if not isinstance(session_history, list):
                continue

            from ..conversation.conversation_utils import format_conversation

            session_str = format_conversation(session_history)
            snippet_len = 1500

            # Take beginning and end if too long
            session_snippet = (
                session_str[: snippet_len // 2] + "\n...\n" + session_str[-snippet_len // 2 :]
                if len(session_str) > snippet_len
                else session_str
            )

            # Add snippet if within total length limit
            if total_snippet_length + len(session_snippet) < max_total_snippet_length:
                snippets.append(f"\n--- Snippet from Session {i + 1} ---\n{session_snippet}")
                total_snippet_length += len(session_snippet)
            else:
                break  # Stop if limit reached

    conversation_snippets = "\n".join(snippets) or "No conversation history available."

    # Choose the appropriate prompt based on chatbot type
    if chatbot_type == "transactional":
        print("   Using TRANSACTIONAL structuring prompt.")
        structuring_prompt = create_transactional_prompt(func_list_str, conversation_snippets)
    else:  # Default to informational
        print("   Using INFORMATIONAL structuring prompt.")
        structuring_prompt = create_informational_prompt(func_list_str, conversation_snippets)

    try:
        print("   Asking LLM to determine workflow structure...")
        response = llm.invoke(structuring_prompt)
        response_content = response.content

        # Extract JSON from the LLM response
        json_str = extract_json_from_response(response_content)

        # Clean up potential JSON issues
        json_str = re.sub(r"//.*?(\n|$)", "\n", json_str)  # Remove comments
        json_str = re.sub(r",(\s*[\]}])", r"\1", json_str)  # Remove trailing commas

        # Parse the JSON string into a list of node info dicts
        structured_nodes_info = json.loads(json_str)
        if not isinstance(structured_nodes_info, list):
            raise ValueError("LLM response is not a JSON list.")

        # Build the hierarchy from the parent_names info
        root_nodes_dicts = build_node_hierarchy(structured_nodes_info)
        print(f"   Built structure with {len(root_nodes_dicts)} root node(s).")

        return root_nodes_dicts

    except json.JSONDecodeError as e:
        print(f"   Error: Failed to decode JSON from LLM response: {e}")
        return flat_functionality_dicts  # Return original list on failure
    except Exception as e:
        print(f"   Error during structure building: {e}")
        return flat_functionality_dicts  # Return original list on failure


def create_transactional_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for transactional chatbot analysis."""
    return f"""
    You are a Workflow Dependency Analyzer. Analyze the discovered interaction steps (functionalities) and conversation snippets to model the **sequential workflow** a user follows.

    Input Functionalities (Extracted Steps):
    {func_list_str}

    Conversation History Snippets (Context for Flow):
    {conversation_snippets}

    CRITICAL TASK: Determine the sequential flow, including prerequisites, branches, and joins based *primarily on the conversational evidence*. Assume a workflow exists unless proven otherwise.
    - **Sequences:** Identify steps that consistently or logically happen *after* others based on the conversation flow (e.g., selecting size after choosing pizza type).
    - **Branches:** Identify points where the chatbot explicitly offers mutually exclusive choices leading to different subsequent steps (e.g., predefined vs. custom pizza).
    - **Joins:** Identify points where different interaction paths converge to the *same* common next step (e.g., adding drinks after either pizza type).

    **IMPORTANT: Distinguish True Prerequisites from Conversational Sequence:**
    - A step should only have `parent_names` if completing the parent step is **functionally required** to perform the child step. Ask: "Is Step A *necessary* to make Step B possible or meaningful?"
    - **Do NOT assign parentage simply because one step occurred before another in a single conversation.**
    - **Meta-Interactions (like asking "What can you do?", "Help", greetings, asking for general info about the bot itself) should almost always be root nodes (`parent_names: []`)**. They describe the interaction *about* the bot, not the core task flow itself. For example, `inquire_main_functionality` or `ask_capabilities` is NOT a prerequisite for `order_pizza`.

    DEEPLY ANALYZE the conversation flow provided:
    1. Which steps seem like entry points? (Potential root nodes, especially meta-interactions)
    2. Which steps are explicitly offered or occur only *after* another specific step is completed **AND are functionally dependent on it**? (Indicates sequence/parent)
    3. Does the chatbot present clear choices followed by different interactions? (Indicates a branch)
    4. Do different paths seem to lead back to the same follow-up step? (Indicates a join)

    Structure the output as a JSON list of nodes. Each node MUST include:
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": List of parameter names (list of strings or []).
    - "parent_names": List of names of functionalities that, based on conversational evidence AND functional necessity, MUST be completed *immediately before* this one (list of strings). Use `[]` for root nodes and meta-interactions.

    Rules for Output:
    - The structure MUST reflect the likely functional dependencies observed in the conversation flow.
    - Use the 'name' field as the identifier.
    - Output MUST be valid JSON. Use [] for empty lists.

    Generate the JSON list representing the precise sequential workflow structure:
    """


def create_informational_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for informational chatbot analysis."""
    return f"""
    You are a Workflow Dependency Analyzer. Analyze the discovered interaction steps (functionalities) and conversation snippets to model the interaction flow, recognizing that it might be **non-sequential Q&A**.

    **CRITICAL CONTEXT:** This chatbot appears primarily **Informational/Q&A**. Users likely ask about independent topics.

    Input Functionalities (Extracted Steps):
    {func_list_str}

    Conversation History Snippets (Context from Multiple Sessions):
    {conversation_snippets}

    CRITICAL TASK: Determine relationships based *only* on **strong conversational evidence ACROSS MULTIPLE SESSIONS**.
    - **Sequences/Branches:** Create parent-child relationships (`parent_names`) ONLY IF the chatbot *explicitly forces* a sequence OR if a step is *impossible* without completing a prior one, AND this dependency is observed CONSISTENTLY.
    - **Independent Topics:** If users ask about different topics independently, treat these functionalities as **separate root nodes** (assign `parent_names: []`). **DO NOT infer dependency just because Topic B was discussed after Topic A in one session.**
    - **Meta-Interactions (like asking "What can you do?", "Help", greetings, asking for general info about the bot itself) should almost always be root nodes (`parent_names: []`)**. They describe the interaction *about* the bot, not the core informational topics themselves.

    **RULE: Your DEFAULT action MUST be to create separate root nodes (empty `parent_names`: `[]`). Only create parent-child links if the conversational evidence for functional dependency is EXPLICIT, CONSISTENT, and UNDENIABLE.** Avoid forcing hierarchies onto informational interactions.

    Structure the output as a JSON list of nodes. Each node MUST include:
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": List of parameter names (list of strings or []).
    - "parent_names": List of names of functionalities that MUST be completed immediately before this one based on the rules above. **Use `[]` for root nodes / independent topics / meta-interactions.**

    Rules for Output:
    - Reflect dependencies (or lack thereof) based STRICTLY on consistent conversational evidence and functional necessity.
    - Use the 'name' field as the identifier.
    - Output MUST be valid JSON. Use [] for empty lists.

    Generate the JSON list representing the interaction flow structure:
    """


def extract_json_from_response(response_content: str) -> str:
    """Extract JSON content from the LLM response."""
    json_str = None
    json_patterns = [
        r"```json\s*([\s\S]+?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]+?)\s*```",  # ``` ... ```
        r"\[\s*\{.*?\}\s*\]",  # Starts with [ { and ends with } ]
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            json_str = match.group(1) if "```" in pattern else match.group(0)
            break

    # Fallback if no pattern matched
    if not json_str:
        if response_content.strip().startswith("[") and response_content.strip().endswith("]"):
            json_str = response_content.strip()
        else:
            raise ValueError("Could not extract JSON block from LLM response.")

    return json_str


def build_node_hierarchy(structured_nodes_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a hierarchical structure from flat nodes with parent references."""
    # Map name to node info
    nodes_map = {node_info["name"]: node_info for node_info in structured_nodes_info if "name" in node_info}

    # Initialize children list for all nodes
    for node_info in nodes_map.values():
        node_info["children"] = []

    # Link children to parents based on 'parent_names'
    for node_name, node_info in nodes_map.items():
        parent_names = node_info.get("parent_names", [])
        for parent_name in parent_names:
            if parent_name in nodes_map:
                parent_node_info = nodes_map[parent_name]
                # Add child if not already present
                if node_info not in parent_node_info.get("children", []):
                    parent_node_info.setdefault("children", []).append(node_info)

    # Find root nodes (nodes that are not children of any other node)
    all_child_names = set()
    for node_info in nodes_map.values():
        for child_info in node_info.get("children", []):
            if isinstance(child_info, dict) and "name" in child_info:
                all_child_names.add(child_info["name"])

    root_nodes = [node_info for node_name, node_info in nodes_map.items() if node_name not in all_child_names]

    return root_nodes
