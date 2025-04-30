from typing import Any


def get_outputs_prompt(
    profile: dict[str, Any],
    sanitized_goals: list[str],
    functionalities: list[str],
    language_instruction: str,
) -> str:
    """Generate the prompt for identifying output fields to extract."""
    return f"""
    Identify 2-4 key pieces of INFORMATION THE CHATBOT WILL PROVIDE that should be extracted from its responses.
    These outputs help the tester validate data from the conversation once it is finished.

    CONVERSATION SCENARIO: {profile.get("name", "Unnamed Profile")}
    USER ROLE: {profile.get("role", "Unknown Role")}
    USER GOALS:
    {", ".join(sanitized_goals)}

    RELEVANT FUNCTIONALITIES:
    {", ".join(functionalities[:5] if functionalities else ["Unknown"])}

    {language_instruction}

    PURPOSE OF OUTPUTS:
    The tester needs to extract specific data points from the chatbot's responses to validate
    the consistency and performance of the chatbot. These outputs represent information
    that should appear in the chatbot's messages.

    EXTREMELY IMPORTANT:
    - Outputs must be information THE CHATBOT GIVES TO THE USER, not what the user inputs
    - Focus on information that validates whether the chatbot is working correctly
    - Choose data points that would appear in the chatbot's responses

    Examples of GOOD outputs:
    - total_cost: The final price quoted by the chatbot for a service
    - appointment_time: When the chatbot says an appointment is available
    - confirmation_number: A reference number provided by the chatbot
    - service_hours: Operating hours mentioned by the chatbot

    Examples of BAD outputs (these are user inputs, not chatbot outputs):
    - selected_toppings: This is what the user tells the chatbot, not what we extract
    - requested_appointment: This is the user's input, not chatbot's output
    - user_query: This is what the user asks, not what we extract

    For each output:
    1. Give it a SHORT, descriptive name (use underscores instead of spaces)
    2. Assign an appropriate type from the following options:
       - str: For text data (names, descriptions, IDs)
       - int: For whole numbers
       - float: For decimal numbers
       - date: For calendar dates
       - time: For time values
    3. Write a brief description of what this output represents and how to find it in the chatbot's responses

    THE OUTPUT STRUCTURE MUST FOLLOW THIS FORMAT:
    OUTPUT: output_name_1
    TYPE: output_type_1
    DESCRIPTION: brief description of what to extract from chatbot responses

    OUTPUT: output_name_2
    TYPE: output_type_2
    DESCRIPTION: brief description of what to extract from chatbot responses

    PROVIDE 2-4 OUTPUTS, EACH FOLLOWING THIS EXACT FORMAT
    """
