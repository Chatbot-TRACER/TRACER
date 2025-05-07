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
5.  **STRICTLY EXCLUDE ALL Meta-Functionalities:** DO NOT extract functionalities that describe:
   - The chatbot introducing itself or explaining its capabilities (e.g., 'introduce_self', 'explain_capabilities', 'list_what_i_can_do')
   - The chatbot asking the user what they want help with (e.g., 'ask_how_to_help', 'greet_user')
   - The user asking what the chatbot can do (e.g., 'ask_capabilities', 'user_requests_help')
   - General self-descriptions or responses to "what can you do?" questions
   - Purely conversational exchanges without specific task-related actions

   **EXCEPTION:** If listing specific, actionable choices is a *required step within a task* (e.g., chatbot lists available service types A, B, C *after* user initiates a request, and the user must choose one to proceed), then THAT specific action (e.g., `present_service_type_options`) IS valid. The key is whether it's a concrete step in a *specific workflow* vs. just a general self-description triggered by "What can you do?".
6.  **Naming:** Use clear, descriptive snake_case names reflecting the *specific* chatbot action or service provided in that step (e.g., `prompt_for_confirmation_details`, `display_search_results`, `initiate_custom_config_flow`).
7.  **Avoid Failures:** AVOID extracting functionalities that solely describe the chatbot failing (e.g., 'handle_fallback', 'fail_to_understand'). Focus on successful actions or information provided.

**CRITICAL RULES FOR PARAMETERS:**
8.  **Parameters are INPUTS to the Chatbot:**
    *   Parameters represent data that the chatbot **explicitly asks the user for** or **needs the user to provide** for the chatbot to perform ITS CURRENT ACTION or complete ITS CURRENT PROMPT.
    *   Think of parameters as the "blanks" the chatbot needs the user to fill in.
9.  **Parameters are NOT Chatbot Outputs:**
    *   If the chatbot is **providing information**, **stating facts**, **confirming something previously provided by the user**, or **presenting options it has generated**, these pieces of information provided *by the chatbot* are NOT parameters *for that information-providing or confirmation action*.
    *   For example:
        *   Chatbot asks "What is your address?" -> Functionality `prompt_for_address` has parameter `address`.
        *   Chatbot says "Your address is 123 Main St." -> Functionality `confirm_address` has NO parameters (123 Main St. is output, not an input for *this* confirmation step).
        *   Chatbot says "Available colors are Red, Green, Blue." -> Functionality `list_available_colors` has NO parameters (Red, Green, Blue are outputs/information provided).
        *   Chatbot asks "Which color do you want: Red, Green, or Blue?" -> Functionality `prompt_for_color_selection` has parameter `selected_color (Red/Green/Blue)`.
10. **Parameter Options:** When the chatbot **solicits an input (a parameter)** AND simultaneously presents specific, limited options for that input (e.g., "What size? We have Small, Medium, Large."), include these options with the parameter. If the chatbot only *presents information* (e.g., "Available sizes are Small, Medium, Large") without an explicit question *in the same turn* for the user to select one, those presented items are NOT parameters for *that* information-providing action.

**EXAMPLES (Focus on Chatbot Actions & Correct Parameter Identification):**

- **Scenario: Booking Appointment (Transactional)**
    - User: "I need to book an appointment."
    - Chatbot: "Okay, what date are you looking for?"
    - User: "Next Tuesday."
    - Chatbot: "For next Tuesday, we have slots at 10 AM and 2 PM available. Which one would you like?"
    - User: "10 AM works."
    - Chatbot: "Great, I need your email to confirm."
    - User: "test@example.com"
    - Chatbot: "Confirmed for Tuesday at 10 AM. Email: test@example.com."
    - **GOOD Extractions (Chatbot Actions & INPUT Parameters):**
        - `prompt_for_booking_date` (Parameters: `booking_date`)
        - `prompt_for_time_slot_selection` (Parameters: `selected_time_slot (10 AM/2 PM)`)
        - `prompt_for_confirmation_email` (Parameters: `confirmation_email`)
        - `confirm_booking_details` (Parameters: None) // Chatbot is outputting/confirming, not asking for input.

- **Scenario: Pizza Ordering (With Options)**
    - User: "I want to order a pizza"
    - Chatbot: "Great! What size would you like? We offer Small, Medium, and Large."
    - User: "Medium"
    - Chatbot: "What toppings would you like? We have Pepperoni, Cheese, Veggie, and Supreme."
    - **GOOD Extraction (With Parameter Options solicited by Chatbot):**
        - `prompt_for_pizza_size` (Parameters: `size (Small/Medium/Large)`)
        - `prompt_for_pizza_toppings` (Parameters: `toppings (Pepperoni/Cheese/Veggie/Supreme)`)

- **Scenario: Providing Options (Granularity)**
    - User: "What are my choices?"
    - Chatbot: "We offer Standard Packages, or you can configure a Custom Solution. Which interests you?"
    - User: "Tell me about Standard Packages."
    - Chatbot: "Our Standard Packages include Basic, Pro, and Enterprise. They offer features X, Y, and Z respectively."
    - **GOOD Granular Extractions (Chatbot Actions & INPUT Parameters):**
        - `prompt_choice_between_standard_or_custom` (Parameters: `package_type_preference (Standard/Custom)`)
        - `describe_standard_packages` (Parameters: None) // Chatbot is providing information.
        - (If user asked about custom) -> `initiate_custom_solution_configuration` (Parameters: None, unless it immediately asks for an input)

- **Scenario: Company Policy Info (Informational)**
    - User: "What's the policy on remote work?"
    - Chatbot: "Our remote work policy allows eligible employees to work remotely up to 3 days per week..."
    - **GOOD Extraction (Chatbot Action - Providing Info):**
        - `explain_remote_work_policy` (Parameters: None) // Chatbot is providing information.

- **Scenario: Meta-Functionality (Should NOT be extracted)**
    - User: "What can you do?"
    - Chatbot: "I can help with ordering food, tracking deliveries, and finding restaurant information."
    - **DO NOT EXTRACT:** `explain_capabilities`, `list_services`, `describe_what_i_can_do`, etc.

- **Scenario: Meta-Functionality vs. Legitimate Choice Menu (Differentiate)**
    - User: "Order food"
    - Chatbot: "What type of food would you like to order? We offer: 1) Pizza, 2) Burgers, 3) Sushi"
    - **VALID Extraction:** `present_food_category_options` (Parameters: `selected_food_category (Pizza/Burgers/Sushi)`) - This is a specific step in a food ordering workflow

For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
1. A specific, descriptive name (snake_case, from chatbot's perspective)
2. A clear description (what the chatbot DOES or PROVIDES in this specific functionality/step)
3. Required parameters: List **ONLY data that the chatbot explicitly solicits from the user as input for THIS specific action/prompt**. If the chatbot is confirming data or just providing information, there are no parameters for that action.
4. Parameter options: If, when soliciting a parameter, the chatbot explicitly presents limited choices for that specific input, include them.

Format EXACTLY as:
FUNCTIONALITY:
name: chatbot_specific_action_name
description: What the chatbot specifically does or provides in this functionality.
parameters: param1 (option1/option2/option3): Brief description of what param1 represents, param2: Description of param2, param3 (optionA/optionB): Description of param3

For parameters without specific options, include a description after a colon.
If there are no parameters for the action (i.e., the chatbot is providing information, confirming details, or asking a question that doesn't require specific data input with options), write "None".

If no new relevant **chatbot capability/action** fitting these criteria is identified in the latest exchanges, respond ONLY with "NO_NEW_FUNCTIONALITY".
"""
