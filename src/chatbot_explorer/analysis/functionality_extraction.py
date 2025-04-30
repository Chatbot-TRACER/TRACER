import re

from chatbot_explorer.conversation.conversation_utils import format_conversation
from chatbot_explorer.schemas.functionality_node import FunctionalityNode


def extract_functionality_nodes(conversation_history, llm, current_node=None) -> list[FunctionalityNode]:
    """Find out FunctionalityNodes from the conversation.

    Args:
        conversation_history (list): The list of chat messages.
        llm: The language model instance.
        current_node (FunctionalityNode, optional): The node being explored. Defaults to None.

    Returns:
        List[FunctionalityNode]: A list of newly found FunctionalityNode objects.
    """
    # Format conversation for the LLM
    formatted_conversation = format_conversation(conversation_history)

    # Context for the LLM
    context = "Identify distinct interaction steps or functionalities the chatbot provides in this conversation, relevant to the user's workflow."
    if current_node:
        context += f"\nWe are currently exploring the '{current_node.name}' step: {current_node.description}"

    # Prompt for the LLM
    extraction_prompt = f"""
    {context}

    CONVERSATION:
    {formatted_conversation}

    Analyze the conversation and extract distinct **chatbot capabilities, actions performed, or information provided by the chatbot**.

    **CRITICAL RULES:**
    1.  **Focus ONLY on the CHATBOT:** Functionalities MUST represent actions performed *by the chatbot* or information *provided by the chatbot*.
    2.  **EXCLUDE User Actions:** DO NOT extract steps that only describe what the *user* asks, says, or does (e.g., 'user_asks_for_help', 'user_selects_option', 'user_provides_details').
    3.  **Chatbot's Perspective:** If a user action triggers a chatbot response or workflow, name and describe the functionality based on the **chatbot's role** in handling that trigger (e.g., `handle_order_request`, `collect_user_selection`, `prompt_for_details`, `provide_requested_info`, `confirm_details`).
    4.  **Naming:** Use clear, descriptive snake_case names reflecting the chatbot's action (e.g., `provide_menu`, `collect_size`, `confirm_order`, `process_payment`).
    5.  **Avoid Failures:** AVOID extracting functionalities that solely describe the chatbot failing to understand or provide information (e.g., 'handle_repeat_requests', 'fail_to_provide_info'). Focus on successful actions or information provided, or capabilities the chatbot *claims* to have.

    **EXAMPLES (Focus on Chatbot Actions):**

    - **Scenario: Booking Appointment (Transactional)**
        - User: "I need to book an appointment."
        - Chatbot: "Okay, what date are you looking for?"
        - User: "Next Tuesday."
        - Chatbot: "We have slots at 10 AM and 2 PM available."
        - User: "10 AM works."
        - Chatbot: "Great, I need your email to confirm."
        - User: "test@example.com"
        - Chatbot: "Confirmed for Tuesday at 10 AM. Email: test@example.com."

        - **BAD Extractions (User Actions):**
            - `user_requests_booking` (User action)
            - `user_provides_date` (User action)
            - `user_selects_time` (User action)
            - `user_gives_email` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `prompt_for_booking_date` (Chatbot asks for date)
            - `provide_available_time_slots` (Chatbot lists options)
            - `prompt_for_confirmation_email` (Chatbot asks for email)
            - `confirm_booking_details` (Chatbot confirms the final booking)

    - **Scenario: Company Policy Info (Informational)**
        - User: "What's the policy on remote work?"
        - Chatbot: "Our remote work policy allows eligible employees to work remotely up to 3 days per week, subject to manager approval. More details are in the employee handbook."
        - User: "Where can I find the handbook?"
        - Chatbot: "You can find the employee handbook on the company intranet under HR Documents."

        - **BAD Extractions (User Actions):**
            - `user_asks_remote_policy` (User action)
            - `user_asks_for_handbook_location` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `explain_remote_work_policy` (Chatbot provides policy summary)
            - `provide_handbook_location` (Chatbot tells where to find the handbook)

    - **Scenario: General Inquiry (Informational/Meta)**
        - User: "What can you do?"
        - Chatbot: "I can help you book appointments, check order status, and answer questions about our services."

        - **BAD Extractions (User Actions):**
            - `user_asks_capabilities` (User action)
        - **GOOD Extractions (Chatbot Actions):**
            - `list_main_capabilities` (Chatbot lists what it can do)


    For each relevant **chatbot capability/action/information provided** based on these rules and examples, identify:
    1. A short, descriptive name (snake_case, from chatbot's perspective).
    2. A clear description (from chatbot's perspective - what the chatbot DOES or PROVIDES).
    3. Required parameters (inputs the chatbot NEEDS for this step, comma-separated or "None").

    Format EXACTLY as:
    FUNCTIONALITY:
    name: chatbot_action_name
    description: What the chatbot does or provides.
    parameters: param1, param2 (or "None")

    If no new relevant **chatbot capability/action** is identified, respond ONLY with "NO_NEW_FUNCTIONALITY".
    """
    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    print("\n--- Raw LLM Response for Functionality Extraction ---")
    print(content)
    print("-----------------------------------------------------")

    # Parse the LLM response
    functionality_nodes = []

    if "NO_NEW_FUNCTIONALITY" in content.upper():  # Case-insensitive check
        print("  LLM indicated no new functionalities.")
        return functionality_nodes

    # Split response into blocks
    blocks = re.split(r"FUNCTIONALITY:\s*", content, flags=re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block:  # Skip empty parts
            continue

        name = None
        description = None
        params_str = "None"

        # Parse lines in the block
        lines = block.split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("name:"):
                name = line[len("name:") :].strip()
            elif line.lower().startswith("description:"):
                description = line[len("description:") :].strip()
            elif line.lower().startswith("parameters:"):
                params_str = line[len("parameters:") :].strip()

        # Create node if we got name and description
        if name and description:
            # Parse parameters string
            parameters = []
            if params_str.lower() != "none":
                param_names = [p.strip() for p in params_str.split(",") if p.strip()]
                # Basic parameter structure
                parameters = [{"name": p, "type": "string", "description": f"Parameter {p}"} for p in param_names]

            new_node = FunctionalityNode(
                name=name,
                description=description,
                parameters=parameters,
                parent=current_node,  # Set parent for now
            )
            functionality_nodes.append(new_node)
            print(f"  Identified step (Robust Parsing): {name}")
        elif block:  # Log blocks we couldn't parse
            print(f"  WARN: Could not parse functionality block:\n{block}")

    if not functionality_nodes and "NO_NEW_FUNCTIONALITY" not in content.upper():
        print("  WARN: LLM response did not contain 'NO_NEW_FUNCTIONALITY' but no functionalities were parsed.")

    return functionality_nodes
