"""Prompts for extracting functionality nodes from a conversation."""


def get_functionality_extraction_prompt(context: str, formatted_conversation: str) -> str:
    """Generate the prompt for extracting functionality nodes from a conversation."""
    return f"""
{context}

CONVERSATION:
{formatted_conversation}

Analyze the conversation to extract distinct **chatbot capabilities, actions performed, or information provided BY THE CHATBOT**. These represent concrete steps towards achieving a user goal or delivering specific information.

**CORE PRINCIPLES FOR EXTRACTION:**
1.  **CHATBOT-CENTRIC:** Functionalities MUST represent actions performed *by the chatbot* or information *provided by the chatbot*.
2.  **EXCLUDE USER ACTIONS:** DO NOT extract steps describing only what the *user* asks, says, or does.
3.  **SPECIFIC & ACTIONABLE:** Aim for concrete, granular actions the chatbot performs within a potential workflow, not abstract categories.
4.  **DIFFERENTIATE PATHS:** If the chatbot offers distinct choices or information categories leading to different interaction paths (e.g., standard vs. custom items), extract SEPARATE functionalities for each distinct path offered by the chatbot.
5.  **AVOID PURELY META-FUNCTIONALITIES:** Do NOT extract functionalities that SOLELY describe the chatbot's general abilities abstractly (e.g., 'list_capabilities', 'explain_what_i_can_do').
    *   **EXCEPTION:** If listing specific, actionable choices is a *required step within a user task* (e.g., chatbot lists service types A, B, C *after* user initiates a request, and user *must* choose one to proceed), then that specific action (e.g., `present_service_type_options`) IS valid.
6.  **NAMING CONVENTION & GENERALIZATION:**
    *   Use clear, descriptive snake_case names reflecting the *specific core action* performed by the chatbot (e.g., `prompt_for_item_size`, `confirm_selection_and_proceed`).
    *   **AIM FOR GENERALITY IN NAME AND DESCRIPTION:** Even if the specific instance in the conversation refers to a "large item" or "next Tuesday," if the chatbot's underlying capability is to "prompt for item size" or "confirm a date," try to name and describe the functionality in those more general terms. The specific values (like "large" or "Tuesday") might be parameters or part of the conversational context but not necessarily part of the core function's name.
    *   Example: If chatbot says "Okay, for the large item, what color do you want?", extract `prompt_for_item_color` (parameter: `item_color`), not `prompt_for_large_item_color`. The "large" context is noted but the function is about getting a color.
7.  **FOCUS ON SUCCESS:** Extract successful actions or information provided. Avoid functionalities that solely describe chatbot failures (e.g., 'handle_fallback').

**RULES FOR IDENTIFYING PARAMETERS (User Inputs Solicited by Chatbot):**
8.  **DEFINITION:** Parameters are data that the chatbot **explicitly asks the user for** or **needs the user to provide** for the chatbot to perform ITS CURRENT ACTION or complete ITS CURRENT PROMPT.
9.  **OPTIONS FOR PARAMETERS:** If the chatbot, when soliciting an input, simultaneously presents specific, limited options for that input (e.g., "What size? We have Small, Medium, Large."), these options (Small, Medium, Large) MUST be included with that parameter. The core is the *solicitation of input alongside presented choices for that input*.

**RULES FOR IDENTIFYING OUTPUTS (Information Provided by Chatbot):**
10. **DEFINITION:** Outputs represent specific pieces of information, data fields, confirmations, or results that the chatbot **provides, states, or displays to the user.** These are typically results of a query or transaction (e.g., an order ID, total price, policy details).
11. **OUTPUT CATEGORIES & DESCRIPTIONS:** When the chatbot presents information, capture this as output categories with concise descriptions of the *type* of information provided (e.g., `service_packages: A range of service tiers from Basic to Premium`). Focus on the NATURE of information, not specific instance values from the conversation (unless Rule 12 applies).
12. **OUTPUTS PRESENTING A LIST OF CHOICES:** If an output's primary purpose is to present a *list of multiple, distinct items or choices* that the user can see (e.g., "Our services are: A, B, C"), then the specific items (A, B, C) should be part of the output description or category value.
    *   Example: `output_options: available_services_list: Services offered include Standard, Premium, and Express.`

**HANDLING INTERACTIONS INVOLVING OPTIONS AND SELECTIONS:**
13. **DISTINGUISHING SINGLE vs. SEPARATE ACTIONS:**
    *   **Case A (Single, Combined Action):** If the chatbot presents a list of options AND *immediately in the same turn* asks for a selection from that list (e.g., "We have Red, Green, Blue. Which color do you pick?"), extract a **single functionality** (e.g., `prompt_for_color_selection_from_list`).
        *   **Parameters:** The presented options (Red, Green, Blue) become `options` for the relevant parameter (e.g., `selected_color (Red/Green/Blue)`).
        *   **Outputs:** This single functionality likely has minimal or no `output_options` itself, as its main goal is to solicit input.
    *   **Case B (Separate, Sequential Actions):** If presenting options and asking for a selection are **clearly distinct turns or chatbot actions** (e.g., Chatbot Turn 1: "Here are our main categories: X, Y, Z." Chatbot Turn 2: "Great, which category would you like to explore?"), extract them as **two separate functionalities**:
        1.  An information-providing functionality (e.g., `present_main_categories`) with `output_options` describing the categories presented (per Rule 11 & 12, e.g., `main_category_options: Describes categories X, Y, and Z`). This function has no parameters.
        2.  A subsequent input-soliciting functionality (e.g., `prompt_for_category_selection`) with a parameter for the user's choice (e.g., `selected_category (X/Y/Z)`). This function has no `output_options`.

**EXAMPLES:**

- **Scenario: Item Ordering (Illustrating Rule 13, Case A - Single Action)**
    - User: "I want to order an item."
    - Chatbot: "Okay! We have Model A, Model B, and Model C. Which model would you like?"
    - User: "Model B."
    - Chatbot: "Great. For Model B, what size? Small, Medium, or Large?"
    - User: "Large."
    - Chatbot: "Okay, 1 Large Model B. Your order ID is 789XYZ. Total is $50. It will be ready in 20 minutes at our Main St location."
    - **GOOD Extractions:**
        - `prompt_for_model_selection_from_list`
            - description: Presents available models and prompts the user to select one in the same turn.
            - parameters: `selected_model (Model A/Model B/Model C)`
            - output_options: None
        - `prompt_for_size_selection_from_list`
            - description: Presents available sizes for the chosen model and prompts the user to select one in the same turn.
            - parameters: `selected_size (Small/Medium/Large)`
            - output_options: None
        - `provide_order_confirmation_and_details`
            - description: Confirms the order and provides the order ID, total cost, estimated readiness time, and pickup location.
            - parameters: None
            - output_options: `ordered_item_summary: Details of the confirmed item; order_identifier: Unique ID for the order; total_cost: Final price; estimated_readiness_time: Expected time for order completion; pickup_location: Store address for pickup`

- **Scenario: Service Inquiry (Illustrating Rule 13, Case B - Separate Actions)**
    - User: "What services do you offer?"
    - Chatbot: "We offer Standard Servicing, Premium Repair, and Express Diagnostics." (Action 1)
    - User: "Tell me more about Premium Repair."
    - Chatbot: "Okay. Which aspect of Premium Repair interests you: features or pricing?" (Action 2 - assuming user needs to choose before details are given)
    - **GOOD Extractions:**
        - `list_available_services` (from Action 1)
            - description: Lists the main types of services offered.
            - parameters: None
            - output_options: `service_types_offered: Standard Servicing, Premium Repair, Express Diagnostics`
        - `prompt_for_premium_repair_detail_type` (from Action 2)
            - description: Asks the user which aspect of Premium Repair they want details about.
            - parameters: `detail_type_preference (features/pricing)`
            - output_options: None

For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
1. A specific, descriptive name (snake_case).
2. A clear description.
3. Required parameters (inputs the chatbot SOLICITS). List the parameter name, and if the chatbot presented explicit choices for that input, list them in parentheses.
4. Output options: For information PROVIDED by the chatbot, list `category_name: description_of_category_content`. Separate multiple categories with a semicolon.

Format EXACTLY as:
FUNCTIONALITY:
name: chatbot_specific_action_name
description: What the chatbot specifically does or provides in this functionality.
parameters: param1 (option1/option2/option3): meaningful description of what param1 represents, param2: description of param2
output_options: category1: description1; category2: description2

If there are no parameters, write "None".
If there are no output options (e.g., the chatbot only asks for input per Rule 13 Case A), write "None".

If no new relevant **chatbot capability/action** fitting these criteria is identified, respond ONLY with "NO_NEW_FUNCTIONALITY".
"""
