"""Prompts for extracting functionality nodes from a conversation."""


def get_functionality_extraction_prompt(context: str, formatted_conversation: str) -> str:
    """Generate the prompt for extracting functionality nodes from a conversation."""
    return f"""
{context}

CONVERSATION:
{formatted_conversation}

Analyze the conversation and extract distinct **chatbot capabilities, actions performed, or information provided by the chatbot** that represent concrete steps towards achieving a user goal or delivering specific information.

**CRITICAL RULES:**
1.  **Focus ONLY on the CHATBOT:** Functionalities MUST represent actions performed *by the chatbot* or information *provided by the chatbot*.
2.  **EXCLUDE User Actions:** DO NOT extract steps that only describe what the *user* asks, says, or does (e.g., 'user_asks_for_help', 'user_selects_option').
3.  **Extract Specific & Actionable Steps:** Aim for concrete actions the chatbot performs within a workflow, not abstract categories.
4.  **Differentiate Based on Paths/Options:** If the chatbot presents distinct *types* or *categories* of options/information that lead to different interaction paths (e.g., standard vs. custom items, different service types), extract SEPARATE functionalities reflecting the specific action for *each distinct path/category offered*. For example, instead of a single `provide_options`, extract `present_standard_item_options` AND `offer_custom_item_configuration` if the chatbot makes that distinction clear.
5.  **EXCLUDE Purely Meta-Functionalities:** Do NOT extract functionalities that SOLELY describe the chatbot's general abilities IN THE ABSTRACT (e.g., 'list_capabilities', 'explain_what_i_can_do', 'state_purpose'). **EXCEPTION:** If listing specific, actionable choices is a *required step within a task* (e.g., chatbot lists available service types A, B, C *after* user initiates a request, and the user must choose one to proceed), then THAT specific action (e.g., `present_service_type_options`) IS valid. The key is whether it's a concrete step in a *specific workflow* vs. just a general self-description triggered by "What can you do?".
6.  **Naming:** Use clear, descriptive snake_case names reflecting the *specific* chatbot action or service provided in that step (e.g., `prompt_for_confirmation_details`, `display_search_results`, `initiate_custom_config_flow`).
7.  **Avoid Failures:** AVOID extracting functionalities that solely describe the chatbot failing (e.g., 'handle_fallback', 'fail_to_understand'). Focus on successful actions or information provided.
8.  **Parameter Options:** If the chatbot presents specific options for parameters (e.g., sizes: small/medium/large, payment methods: cash/credit), include these options with the parameter.

**EXAMPLES (Focus on Chatbot Actions & Granularity):**

- **Scenario: Booking Appointment (Transactional)**
    - User: "I need to book an appointment."
    - Chatbot: "Okay, what date are you looking for?"
    - User: "Next Tuesday."
    - Chatbot: "We have slots at 10 AM and 2 PM available."
    - User: "10 AM works."
    - Chatbot: "Great, I need your email to confirm."
    - User: "test@example.com"
    - Chatbot: "Confirmed for Tuesday at 10 AM. Email: test@example.com."
    - **GOOD Extractions (Chatbot Actions):**
        - `prompt_for_booking_date`
        - `provide_available_time_slots` with parameters: time_slot (10 AM/2 PM)
        - `prompt_for_confirmation_email`
        - `confirm_booking_details`

- **Scenario: Pizza Ordering (With Options)**
    - User: "I want to order a pizza"
    - Chatbot: "Great! What size would you like? We offer Small, Medium, and Large."
    - User: "Medium"
    - Chatbot: "What toppings would you like? We have Pepperoni, Cheese, Veggie, and Supreme."
    - **GOOD Extraction (With Parameter Options):**
        - `prompt_for_pizza_size` with parameters: size (Small/Medium/Large)
        - `prompt_for_pizza_toppings` with parameters: toppings (Pepperoni/Cheese/Veggie/Supreme)

- **Scenario: Providing Options (Granularity)**
    - User: "What are my choices?"
    - Chatbot: "We offer Standard Packages, or you can configure a Custom Solution. Which interests you?"
    - User: "Tell me about Standard Packages."
    - Chatbot: "Our Standard Packages include Basic, Pro, and Enterprise..."
    - **GOOD Granular Extractions (Chatbot Actions):**
        - `offer_choice_between_standard_or_custom` with parameters: package_type (Standard/Custom)
        - `describe_standard_packages` with parameters: package (Basic/Pro/Enterprise)
        - (If user asked about custom) -> `initiate_custom_solution_configuration`

- **Scenario: Company Policy Info (Informational)**
    - User: "What's the policy on remote work?"
    - Chatbot: "Our remote work policy allows eligible employees to work remotely up to 3 days per week..."
    - **GOOD Extraction (Chatbot Action):**
        - `explain_remote_work_policy`

For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
1. A specific, descriptive name (snake_case, from chatbot's perspective)
2. A clear description (what the chatbot DOES or PROVIDES in this specific functionality/step)
3. Required parameters (inputs the chatbot NEEDS for this step)
4. Parameter options (if the chatbot explicitly presents limited choices for inputs)

Format EXACTLY as:
FUNCTIONALITY:
name: chatbot_specific_action_name
description: What the chatbot specifically does or provides in this functionality.
parameters: param1 (option1/option2/option3), param2, param3 (optionA/optionB)

For parameters without specific options, just list the parameter name. For parameters with options, include them in parentheses with forward slashes between options.
If there are no parameters, just write "None".

If no new relevant **chatbot capability/action** fitting these criteria is identified in the latest exchanges, respond ONLY with "NO_NEW_FUNCTIONALITY".
"""
