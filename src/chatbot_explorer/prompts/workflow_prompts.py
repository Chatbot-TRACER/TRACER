"""Prompts for workflow analysis and modeling of chatbot interactions."""


def create_transactional_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for transactional chatbot analysis."""
    return f"""
    You are a Workflow Dependency Analyzer. Analyze the discovered interaction steps (functionalities) and conversation snippets to model the **sequential workflow** a user follows.

    Input Functionalities (Extracted Steps):
    {func_list_str}

    Conversation History Snippets (Context for Flow):
    {conversation_snippets}

    CRITICAL TASK: Determine the sequential flow, including prerequisites, branches, and joins based *primarily on the conversational evidence*. Assume a workflow exists unless proven otherwise.
    - **Sequences:** Identify steps that consistently or logically happen *after* others based on the conversation flow (e.g., selecting a specific option like size or color *after* choosing a general product type).
    - **Branches:** Identify points where the chatbot explicitly offers mutually exclusive choices leading to different subsequent steps (e.g., choosing between standard service options vs. a customized request).
    - **Joins:** Identify points where different interaction paths converge to the *same* common next step (e.g., confirming contact details after either requesting a quote or reporting an issue).

    **IMPORTANT: Distinguish True Prerequisites from Conversational Sequence:**
    - A step should only have `parent_names` if completing the parent step is **functionally required** to perform the child step. Ask: "Is Step A *necessary* to make Step B possible or meaningful?"
    - **Do NOT assign parentage simply because one step occurred before another in a single conversation.**
    - **Meta-Interactions (like asking "What can you do?", "Help", greetings, asking for general info about the bot itself) should almost always be root nodes (`parent_names: []`)**. They describe the interaction *about* the bot, not the core task flow itself. For example, `inquire_main_functionality` or `ask_capabilities` is NOT a prerequisite for `complete_primary_task` (like placing an order or submitting a request).

    DEEPLY ANALYZE the conversation flow provided:
    1. Which steps seem like entry points? (Potential root nodes, especially meta-interactions)
    2. Which steps are explicitly offered or occur only *after* another specific step is completed **AND are functionally dependent on it**? (Indicates sequence/parent)
    3. Does the chatbot present clear choices followed by different interactions? (Indicates a branch)
    4. Do different paths seem to lead back to the same follow-up step? (Indicates a join)

    Structure the output as a JSON list of nodes. Each node MUST include:
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": List of parameter objects. Each parameter object should include:
        - "name": Parameter name (string).
        - "description": Parameter description (string).
        - "options": List of available options for the parameter (list of strings or [] if no specific options).
      If a functionality has no parameters, use an empty list `[]`.
    - "outputs": List of output objects. Each output object should include:
        - "category": Category name (string).
        - "description": Description of the output (string).
      If a functionality has no outputs, use an empty list `[]`.
    - "parent_names": List of names of functionalities that, based on conversational evidence AND functional necessity, MUST be completed *immediately before* this one (list of strings). Use `[]` for root nodes and meta-interactions.

    Rules for Output:
    - The structure MUST reflect the likely functional dependencies observed in the conversation flow.
    - Use the 'name' field as the identifier.
    - IMPORTANT: Preserve ALL parameters AND outputs from the input functionalities.
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
    - "parameters": List of parameter objects. Each parameter object should include:
        - "name": Parameter name (string).
        - "description": Parameter description (string).
        - "options": List of available options for the parameter (list of strings or [] if no specific options).
      If a functionality has no parameters, use an empty list `[]`.
    - "outputs": List of output objects. Each output object should include:
        - "category": Category name (string).
        - "description": Description of the output (string).
      If a functionality has no outputs, use an empty list `[]`.
    - "parent_names": List of names of functionalities that MUST be completed immediately before this one based on the rules above. **Use `[]` for root nodes / independent topics / meta-interactions.**

    Rules for Output:
    - Reflect dependencies (or lack thereof) based STRICTLY on consistent conversational evidence and functional necessity.
    - Use the 'name' field as the identifier.
    - IMPORTANT: Preserve ALL parameters AND outputs from the input functionalities.
    - Output MUST be valid JSON. Use [] for empty lists.

    Generate the JSON list representing the interaction flow structure:
    """
