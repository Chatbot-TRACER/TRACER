"""Prompts for extracting functionality nodes from a conversation."""


def get_functionality_extraction_prompt(context: str, formatted_conversation: str) -> str:
    """Generate the prompt for extracting functionality nodes from a conversation."""
    return f"""
{context}

CONVERSATION:
{formatted_conversation}

Analyze the conversation and extract distinct **chatbot capabilities, actions performed, or information provided by the chatbot** that represent concrete steps towards achieving a user goal or delivering specific information.

**CRITICAL RULES (General):**
1.  **Focus ONLY on the CHATBOT:** Functionalities MUST represent actions performed *by the chatbot* or information *provided by the chatbot*.
2.  **EXCLUDE User Actions:** DO NOT extract steps that only describe what the *user* asks, says, or does.
3.  **Extract Specific & Actionable Steps:** Aim for concrete actions.
4.  **Differentiate Paths:** If distinct paths/options are offered leading to different outcomes, extract separate functionalities.
5.  **EXCLUDE Purely Meta-Functionalities** (e.g., 'list_capabilities') unless a required step in a task.
6.  **Naming:** Clear, snake_case names (e.g., `prompt_for_delivery_address`, `display_order_summary`).
7.  **Avoid Failures:** Focus on successful actions.

**CRITICAL RULES FOR PARAMETERS (INPUTS SOLICITED BY CHATBOT):**
8.  **Parameters are What the Chatbot ASKS FOR:** Parameters represent data the chatbot **explicitly asks the user for** or **needs the user to provide** to complete ITS CURRENT PROMPT or action.
9.  **Parameter Options are Presented Choices for Input:**
    *   If the chatbot says, "Which X do you want? We offer A, B, C." then the functionality is `prompt_for_X_selection`, and its parameter is `selected_X` with options `(A/B/C)`.
    *   The list (A, B, C) provided by the chatbot *becomes the options for the parameter it is soliciting*.
    *   This is true even if the "presenting options" and "asking for choice" happen in the same chatbot turn. The core is the *solicitation of input*.

**CRITICAL RULES FOR OUTPUTS (INFORMATION PROVIDED BY CHATBOT):**
10. **Outputs are What the Chatbot GIVES/STATES (not what it asks for):**
    *   Outputs represent specific pieces of information, data fields, confirmations, or results that the chatbot **provides, states, or displays to the user.**
    *   These are typically things the user would expect to see as a result of their query or a completed transaction (e.g., an order ID, a total price, a delivery ETA, requested policy details, a weather forecast).
11. **Output Options are Information PROVIDED by the Chatbot:**
    * Output options represent specific categories of information that the chatbot presents to the user.
    * When the chatbot presents information, capture these as output categories with descriptions.
    * **IMPORTANT:** Outputs represent what the chatbot GIVES, while parameters represent what the chatbot ASKS FOR.
    * For example:
        * Chatbot says "Our service packages include Basic, Standard, and Premium" -> Output options are "service_packages: A range of service tiers from Basic to Premium"
        * Chatbot presents "Available service tiers: Basic ($50/mo), Standard ($75/mo), Premium ($100/mo)" -> Output options are "pricing_information: Monthly subscription costs for different service tiers"
        * Chatbot says "Your transaction ID is XZ987." -> Output options are "transaction_details: Unique identifier for the current transaction"
        * Chatbot says "We are located at 123 Main St." -> Output options are "location_information: Physical address of the business"
12. **Conceptual Difference from Parameters:**
    * When the chatbot PRESENTS information (outputs) and then ASKS for a selection from that information, you must capture BOTH:
        * The presented information as OUTPUT OPTIONS for the information-providing functionality
        * The user selection as a PARAMETER for the follow-up choice-requesting functionality
    * Example:
        * Chatbot: "We offer options A, B, and C. Which option would you like?"
        * This should be captured as TWO related functionalities:
            1. `present_available_options` with OUTPUT OPTIONS "available_options: Description of the options presented to the user" (no parameters)
            2. `prompt_for_option_selection` with PARAMETER "selected_option (A/B/C)"
13. **ALWAYS Capture Information-Only Outputs:**
    * For informational chatbots that primarily provide facts, data, or options without requesting input, focus on capturing detailed output categories.
    * Example:
        * Chatbot: "Our store hours are: Monday-Friday 9am-5pm, Saturday 10am-4pm, Sunday Closed."
        * This should be captured as functionality `provide_store_hours` with OUTPUT OPTIONS "operating_hours: Business hours for different days of the week"
    * Example:
        * Chatbot: "The weather in New York today is 72°F and sunny."
        * This should be captured as functionality `provide_weather_information` with OUTPUT OPTIONS "current_weather: Temperature and conditions for the specified location"
14. **Categorize Output Options:**
    * Create meaningful categories that describe the TYPE of information being provided (e.g., "product_features", "subscription_levels", "available_slots", "pricing_details", "item_attributes")
    * Provide clear, concise descriptions that summarize the information in each category
    * Focus on the NATURE of the information rather than specific values

**EXAMPLES (Focus on Chatbot Actions, Parameters, and new Output Definition):**

- **Scenario: Item Ordering (Transactional)**
    - User: "I want to order an item."
    - Chatbot: "Okay! We have Model A, Model B, and Model C. Which model would you like?"
    - User: "Model B."
    - Chatbot: "Great. For Model B, what size? Small, Medium, or Large?"
    - User: "Large."
    - Chatbot: "Okay, 1 Large Model B. Your order ID is 789XYZ. Total is $50. It will be ready in 20 minutes at our Main St location."
    - **GOOD Extractions:**
        - `prompt_for_model_selection`
            - description: Prompts the user to select a model after presenting available options.
            - parameters: `selected_model (Model A/Model B/Model C)`
            - output_options: None
        - `prompt_for_size_selection`
            - description: Prompts the user to select a size for the chosen model after presenting available sizes.
            - parameters: `selected_size (Small/Medium/Large)`
            - output_options: None
        - `provide_order_confirmation_and_details`
            - description: Confirms the order and provides the order ID, total cost, estimated readiness time, and pickup location.
            - parameters: None
            - output_options: `ordered_item_description_field; order_identifier_field; total_cost_field; estimated_readiness_time_field; pickup_location_address_field`

- **Scenario: Information Request**
    - User: "What are your weekend hours?"
    - Chatbot: "We are open Saturday 10 AM - 6 PM, and Sunday 12 PM - 4 PM."
    - **GOOD Extraction:**
        - `provide_weekend_operating_hours`
            - description: Provides the store's operating hours for Saturday and Sunday.
            - parameters: None
            - output_options: `saturday_hours_info_field; sunday_hours_info_field`

- **Scenario: Booking Confirmation**
    - User: "Book it for 2 PM."
    - Chatbot: "Confirmed! Your appointment is for Tuesday at 2 PM. Your confirmation number is BK-123."
    - **GOOD Extraction:**
        - `confirm_appointment_details`
            - description: Confirms the booked appointment date and time, and provides a confirmation number.
            - parameters: None
            - output_options: `confirmed_appointment_date_field; confirmed_appointment_time_field; appointment_confirmation_number_field`

For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
1. A specific, descriptive name (snake_case).
2. A clear description.
3. Required parameters (inputs the chatbot SOLICITS). List the parameter name, and if the chatbot presented explicit choices for that input, list them in parentheses.
4. Output options: List the snake_case names of the *types* or *categories* of information the chatbot PROVIDES as a result or statement (e.g., `order_summary_details; total_price_amount; user_id_field`). Separate multiple fields with a semicolon.

Format EXACTLY as:
FUNCTIONALITY:
name: chatbot_specific_action_name
description: What the chatbot specifically does or provides in this functionality.
parameters: param1 (option1/option2/option3), param2, param3 (optionA/optionB)
output_options: category1: description1; category2: description2

For parameters without specific options, just list the parameter name.
If there are no parameters for the action (i.e., the chatbot is providing information, confirming details, or asking a question that doesn't require specific data input with options), write "None".
If there are no output options, write "None".

If no new relevant **chatbot capability/action** fitting these criteria is identified, respond ONLY with "NO_NEW_FUNCTIONALITY".
"""
