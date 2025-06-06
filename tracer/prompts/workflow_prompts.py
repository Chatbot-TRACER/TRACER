"""Prompts for workflow analysis and modeling of chatbot interactions."""


def create_transactional_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for transactional chatbot analysis."""
    return f"""
    You are a meticulous Workflow Dependency Analyzer AND Thematic Categorizer. Your primary task is to analyze the provided functionalities (extracted interaction steps) and conversation snippets to model the **precise sequential workflow** a user follows to complete a transaction or achieve a core goal. Your secondary task is to suggest a high-level thematic category for each functionality.

    Input Functionalities (Extracted Steps):
    {func_list_str}

    Conversation History Snippets (Context for Flow):
    {conversation_snippets}

    **TASK 1: Determine Functional Dependencies to Assign `parent_names`**

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

    **TASK 2: Suggest a Thematic Category**

    For EACH functionality, assign a `suggested_category` - a short, descriptive thematic label that groups similar functionalities together. Aim for a small, consistent set of broad categories.

    Examples of good categories for transactional chatbots:
    * "Order Placement" - For steps related to selecting and specifying an order
    * "Payment" - For steps related to payment methods, billing, etc.
    * "User Authentication" - For steps related to login, verification, etc.
    * "Order Confirmation" - For steps confirming or summarizing an order
    * "Chatbot Meta" - For general chatbot interaction like greetings, explanations
    * "Customer Support" - For troubleshooting or help-related functionalities

    Maintain consistency by reusing categories when appropriate rather than creating too many unique ones.

    **ANALYSIS FOCUS:**
    For every functionality, meticulously trace back in the conversation: What was the *chatbot's very last action or prompt* that was *essential* for this current functionality to proceed? That is its parent. If no such single essential step exists, it's likely a root node or its parent is misidentified.

    **OUTPUT STRUCTURE:**
    Return a JSON list where each object represents one of the unique input functionalities, augmented with the determined `parent_names` and `suggested_category`.
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": (Preserve EXACTLY from input).
    - "outputs": (Preserve EXACTLY from input).
    - "parent_names": List of names of functionalities that meet ALL the STRICT criteria above. Use `[]` for root nodes.
    - "suggested_category": A short string for the thematic category (e.g., "Order Placement", "Payment").

    **FINAL INSTRUCTIONS:**
    -   Preserve ALL original details (name, description, parameters, outputs) for each functionality.
    -   The list should contain each input functionality name exactly once.
    -   Focus entirely on deriving the most accurate, functionally necessary `parent_names`.
    -   The `suggested_category` is for organizational purposes and does not influence `parent_names`.
    -   Output valid JSON.

    Generate the JSON list:
    """


def create_informational_prompt(func_list_str: str, conversation_snippets: str) -> str:
    """Create a prompt specifically for informational chatbot analysis."""
    return f"""
    You are a meticulous Workflow Dependency Analyzer AND Thematic Categorizer. Your primary task is to analyze the provided functionalities (interaction steps) and conversation snippets to model the **interaction flow**.
    You MUST recognize that for an **Informational/Q&A chatbot**, most functionalities will likely be **independent topics/root nodes**. Your secondary task is to suggest a high-level thematic category for each functionality.

    **CRITICAL CONTEXT FOR THIS TASK:**
    - This chatbot appears primarily **Informational/Q&A**. Users likely ask about independent topics.
    - **Your DEFAULT action MUST be to assign `parent_names: []` (root node) to each functionality.**
    - ONLY create parent-child links if conversational evidence for a *strict, forced functional dependency* is EXPLICIT, CONSISTENT, and UNDENIABLE.

    Input Functionalities (Name and Description Only):
    {func_list_str}
    # Note: Full Parameter and Output details for each functionality are known but omitted here for brevity.
    # YOU MUST PRESERVE the original parameters and outputs associated with each
    # functionality name when generating the final JSON output. Your task is to determine `parent_names` and suggest a thematic category.

    Conversation History Snippets (Context for Flow):
    {conversation_snippets}

    **TASK 1: Determine Minimal Functional Dependencies to Assign `parent_names`**

    For EACH functionality provided in the input, you MUST determine its `parent_names`. Your **default is `parent_names: []`**.
    Assign a parent (i.e., a non-empty `parent_names` list) ONLY IF Parent P meets **ALL** of the following extremely strict criteria:
    1.  **Explicit Forced Sequence:** The chatbot *explicitly states* or *programmatically forces* the user from Parent P to Child F (e.g., asks a clarifying question P whose answer F is required).
    2.  **Absolute Functional Necessity:** Child F is *literally impossible or completely nonsensical* without the specific information or state created by Parent P immediately preceding it.
    3.  **Consistent Observation (Ideal):** This explicit dependency should ideally be observed consistently whenever these functionalities appear.
    4.  **Closest Functional Link:** Even if a rare dependency exists, only list the *single closest* necessary parent step.

    **RULES FOR `parent_names` ASSIGNMENT (Informational Context):**
    *   **Unique Functionalities:** The output JSON list should contain each unique functionality name from the input ONCE. Your primary task is to assign the correct `parent_names` (defaulting to `[]`).
    *   **OVERWHELMING DEFAULT: Root Nodes (`parent_names: []`):**
        *   Assign `[]` to functionalities representing distinct informational topics (e.g., `provide_opening_hours`, `explain_return_policy`, `describe_product_X_features`).
        *   Assume ALL topics are independent unless an EXPLICIT forced sequence is proven by the conversation.
        *   Meta-interactions (e.g., `greet_user`, `list_capabilities`, `request_rephrasing`) are ALWAYS roots.
        *   If in ANY doubt, assign `parent_names: []`.
    *   **RARE Exceptions for Non-Root Nodes (Potential Parents):**
        *   A node that presents clarification options for a complex topic (e.g., `offer_topic_A_subcategories`) *might* be a parent to a node providing details on a chosen subcategory (`provide_details_for_subcategory_A1`), but ONLY if the chatbot *forces* this selection path.
        *   A node asking for essential identifying information *before* providing specific data (e.g., `prompt_for_policy_document_name`) *might* be a parent to the node providing that specific document (`display_policy_document_X`), if the document cannot be displayed otherwise.
    *   **AVOID Inferring Links:**
        *   Do NOT link `topic_A` to `topic_B` just because a user asked about them sequentially in one conversation.
        *   Do NOT link based on simple topical similarity.
        *   Do NOT link just because one piece of information *could* be useful before another, unless the chatbot *forces* that order.

    **TASK 2: Suggest a Thematic Category**

    For EACH functionality, assign a `suggested_category` - a short, descriptive thematic label that groups similar functionalities together. Aim for a small, consistent set of broad categories.

    Examples of good categories for informational chatbots:
    * "Account & Access" - For account management, login issues, access rights
    * "Network & Connectivity" - For network setup, troubleshooting, connections
    * "Software & Applications" - For software features, updates, compatibility
    * "Hardware Support" - For device-specific information and troubleshooting
    * "General Information" - For company info, policies, etc.
    * "Chatbot Meta" - For chatbot capabilities, help commands, etc.

    Maintain consistency by reusing categories when appropriate rather than creating too many unique ones.

    **ANALYSIS FOCUS:**
    For every functionality, assume `parent_names: []`. Only override this default if you find **undeniable proof** in the conversation snippets of an **explicitly forced sequence** or **absolute functional necessity** linking it directly to an immediate predecessor performed by the chatbot.

    **OUTPUT STRUCTURE:**
    Return a JSON list where each object represents one of the unique input functionalities, augmented with the determined `parent_names` and `suggested_category`.
    - "name": Functionality name (string).
    - "description": Description (string).
    - "parameters": (Preserve EXACTLY from original data - **DO NOT OMIT**).
    - "outputs": (Preserve EXACTLY from original data - **DO NOT OMIT**).
    - "parent_names": List of names of functionalities that meet ALL the STRICT criteria above (this will be `[]` for MOST, if not all, nodes).
    - "suggested_category": A short string for the thematic category (e.g., "Account & Access", "Network & Connectivity").

    **FINAL INSTRUCTIONS:**
    -   Preserve ALL original details (name, description, parameters, outputs) for each functionality in the final JSON.
    -   The list should contain each input functionality name exactly once.
    -   Focus entirely on deriving the most accurate `parent_names`, defaulting STRONGLY to `[]`.
    -   The `suggested_category` is for organizational purposes and does not influence `parent_names`.
    -   Output valid JSON. Ensure the entire response is a single, well-formed JSON list.

    Generate the JSON list:
    """
