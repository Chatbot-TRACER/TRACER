"""Prompts for workflow analysis and modeling of chatbot interactions."""


def create_transactional_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for transactional chatbot analysis."""
    return f"""
    You are a meticulous Workflow Dependency Analyzer. Your task is to analyze the provided functionalities (extracted interaction steps) and conversation snippets to model the **precise sequential workflow** a user follows to complete a transaction or achieve a core goal.

    Input Functionalities (Extracted Steps):
    {func_list_str}

    Conversation History Snippets (Context for Flow):
    {conversation_snippets}

    **CRITICAL TASK: Determine Functional Dependencies to Assign `parent_names`**

    For EACH functionality, you MUST determine its `parent_names`. A functionality (Child F) should have a parent functionality (Parent P) in `parent_names` ONLY IF Parent P meets **ALL** of the following strict criteria:
    1.  **Immediate Functional Prerequisite:** Parent P must be a step that the chatbot performs, or an input the chatbot solicits, that is **DIRECTLY AND IMMEDIATELY NECESSARY** for Child F to occur or make sense. Ask: "Could Child F meaningfully happen if Parent P did not just occur?" If yes, P is NOT a parent.
    2.  **Chatbot-Driven Sequence:** The conversation must show the *chatbot* leading the user from Parent P to Child F, or Child F being a direct response/action taken by the chatbot after Parent P.
    3.  **Avoid Transitive Dependencies as Direct Parents:** If A -> B -> C, then C's immediate parent is B, not A. Do not list A as a direct parent of C unless A *also* has a direct, independent path to C. Focus on the *closest* necessary preceding step.
    4.  **Single Core Task Focus:** Assume the conversation is trying to complete ONE primary user goal (e.g., order an item, book an appointment). Steps should logically connect towards this goal.

    **RULES FOR `parent_names`:**
    *   **Root Nodes:** If a functionality can start an interaction, or is a general greeting/meta-interaction (like "welcome", "list_capabilities"), it should have `parent_names: []`.
        *   Example: `provide_welcome_message` -> `parent_names: []`.
        *   Example: If the first step in ordering is `present_item_categories`, then `present_item_categories` (if not preceded by a general welcome that *forces* it) -> `parent_names: []`.
    *   **Sequential Steps:**
        *   `prompt_for_X` is often a parent to `confirm_X_input`.
        *   `prompt_for_X` is often a parent to `prompt_for_Y` (if Y logically follows X, e.g., prompt for size, then prompt for color).
        *   `present_options_A_B_C` followed by `prompt_for_selection_from_A_B_C` -> `present_options_A_B_C` is parent to `prompt_for_selection_from_A_B_C`.
    *   **Branches:** If `offer_choice_X_or_Y` leads to either `do_action_X` or `do_action_Y`, then `offer_choice_X_or_Y` is a parent to both `do_action_X` and `do_action_Y`.
    *   **Joins:** If `do_action_X` and `do_action_Y` can both lead to a common next step like `confirm_details`, then `confirm_details` would have `parent_names: ["do_action_X", "do_action_Y"]` (if both paths are observed and lead *directly* to it).

    **DEEPLY ANALYZE the conversation flow for functional necessity:**
    1.  **Entry Points:** What are the true starting points of a core task? (Likely `parent_names: []`). `provide_welcome_message` is a classic root. The first *task-specific* prompt (e.g., `prompt_for_item_category`) might also be a root if it's not strictly forced by an earlier step.
    2.  **Dependencies:** For every other step, scrutinize: "What SINGLE step *performed by the chatbot* (or input *solicited by the chatbot*) was ABSOLUTELY ESSENTIAL and happened *just before* this current step, making this current step possible?" That's its parent.
    3.  **User Actions Don't Define Parents:** Chatbot actions are parents to other chatbot actions/prompts.
    4.  **Avoid Over-Linking:** Be conservative. If a step *could* happen independently, even if it often follows another, it might not be a strict child unless functionally dependent.

    Structure the output as a JSON list of nodes. Each node MUST include:
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": List of parameter objects (preserve from input). Each parameter object should include:
        - "name": Parameter name (string).
        - "description": Parameter description (string).
        - "options": List of available options for the parameter (list of strings or [] if no specific options).
      If a functionality has no parameters, use an empty list `[]`.
    - "outputs": List of output objects (preserve from input). Each output object should include:
        - "category": Category name (string).
        - "description": Description of the output (string).
      If a functionality has no outputs, use an empty list `[]`.
    - "parent_names": List of names of functionalities that meet the STRICT criteria above. Use `[]` for root nodes.

    Rules for Output:
    - The `parent_names` MUST reflect **strict, immediate functional dependencies** observed in the conversation flow.
    - Preserve ALL original functionality details (name, description, parameters, outputs). Only `parent_names` are being determined here.
    - Output MUST be valid JSON. Use `[]` for empty lists.

    Generate the JSON list, focusing meticulously on assigning correct `parent_names`:
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
