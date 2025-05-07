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

    For EACH functionality provided in the input, you MUST determine its `parent_names`. A functionality (Child F) should have a parent functionality (Parent P) in `parent_names` ONLY IF Parent P meets **ALL** of the following strict criteria:
    1.  **Immediate & Necessary Prerequisite:** Parent P must be a step (an action or prompt by the chatbot) that is **DIRECTLY AND IMMEDIATELY NECESSARY** for Child F to occur or make logical sense in the conversation. Ask: "Is Child F impossible or nonsensical if Parent P did not *just* happen?" If Child F could happen without P, or if other steps intervene, P is NOT an immediate parent.
    2.  **Chatbot-Driven Sequence:** The conversation flow must clearly show the *chatbot* initiating Parent P, which then directly leads to the chatbot initiating Child F.
    3.  **Closest Functional Link:** If A -> B -> C, then C's immediate parent is B, not A. Focus ONLY on the *single closest* necessary preceding step performed by the chatbot.
    4.  **Core Task Progression:** Assume the conversation aims to complete ONE primary user goal (e.g., order an item). Steps listed as parents must be essential for progressing this core task.

    **RULES FOR `parent_names` ASSIGNMENT:**
    *   **Unique Functionalities:** The output JSON list should contain each unique functionality name from the input ONCE. Your primary task is to assign the correct `parent_names` to these unique functionalities.
    *   **Root Nodes (`parent_names: []`):**
        *   Functionalities that initiate a core task or a distinct sub-flow (e.g., `provide_welcome_message`, `start_new_order_flow`, `request_help_topic_selection`).
        *   General meta-interactions (e.g., `greet_user`, `explain_chatbot_capabilities`) are typically roots.
        *   The first *task-specific* prompt by the chatbot in a clear sequence (e.g., `prompt_for_item_category_to_order`) is often a root if not forced by a preceding meta-interaction.
    *   **Sequential Steps (Common Patterns):**
        *   `prompt_for_X_input` is a strong candidate to be a parent of `confirm_X_input_details`.
        *   `prompt_for_X_input` can be a parent of `prompt_for_Y_input` if Y is the immediate next piece of information solicited by the chatbot in a sequence (e.g., `prompt_for_size` -> `prompt_for_color`).
        *   `present_choices_A_B_C` is a parent of `prompt_for_selection_from_A_B_C` if the prompt immediately follows the presentation of choices by the chatbot.
    *   **Branches:** If `offer_choice_path1_or_path2` leads to either `initiate_path1_action` or `initiate_path2_action`, then `offer_choice_path1_or_path2` is a parent to both.
    *   **Joins:** If distinct paths (e.g., `complete_path1_final_step` and `complete_path2_final_step`) BOTH can directly lead to a common subsequent step (e.g., `display_final_summary`), then `display_final_summary` would have `parent_names: ["complete_path1_final_step", "complete_path2_final_step"]`.
    *   **AVOID Conversational Fluff as Parents:** Steps like `thank_user`, `acknowledge_input_received`, or general empathetic statements are RARELY functional parents. Do NOT list them as parents if a more direct data-gathering step or action-enabling step is the true prerequisite.

    **ANALYSIS FOCUS:**
    For every functionality, meticulously trace back in the conversation: What was the *chatbot's very last action or prompt* that was *essential* for this current functionality to proceed? That is its parent. If no such single essential step exists, it's likely a root node or its parent is misidentified.

    **OUTPUT STRUCTURE:**
    Return a JSON list where each object represents one of the unique input functionalities, augmented ONLY with the determined `parent_names`.
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": (Preserve EXACTLY from input).
    - "outputs": (Preserve EXACTLY from input).
    - "parent_names": List of names of functionalities that meet ALL the STRICT criteria above. Use `[]` for root nodes.

    **FINAL INSTRUCTIONS:**
    -   Preserve ALL original details (name, description, parameters, outputs) for each functionality.
    -   The list should contain each input functionality name exactly once.
    -   Focus entirely on deriving the most accurate, functionally necessary `parent_names`.
    -   Output valid JSON.

    Generate the JSON list:
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
