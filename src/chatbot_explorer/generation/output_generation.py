import re

from chatbot_explorer.constants import VARIABLE_PATTERN

# --- Output Generation Logic ---


def generate_outputs(profiles, functionalities, llm, supported_languages=None):
    """Generate output fields to extract from chatbot responses."""
    # Work in the detected primary language
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Write the descriptions in {primary_language}."

    for profile in profiles:
        # Replace variables in goals with general terms for better LLM understanding
        sanitized_goals = []
        for goal in profile.get("goals", []):
            # Replace {{variable}} with general terms
            if isinstance(goal, str):
                sanitized_goal = re.sub(VARIABLE_PATTERN, "[specific details]", goal)
                sanitized_goals.append(sanitized_goal)

        outputs_prompt = f"""
        Identify 2-4 key pieces of INFORMATION THE CHATBOT WILL PROVIDE that should be extracted from its responses.
        These outputs help the tester validate data from the conversation once it is finished.

        CONVERSATION SCENARIO: {profile["name"]}
        USER ROLE: {profile["role"]}
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

        outputs_response = llm.invoke(outputs_prompt)

        # Parse the outputs
        outputs_list = []
        current_output = None
        current_data = {}

        for line in outputs_response.content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("OUTPUT:"):
                # Save previous output if exists
                if current_output and current_data:
                    outputs_list.append({current_output: current_data})

                # Start new output
                current_output = line[len("OUTPUT:") :].strip()
                # Ensure name has no spaces and is lowercase
                current_output = current_output.replace(" ", "_").lower()
                current_data = {}

            elif line.startswith("TYPE:"):
                current_data["type"] = line[len("TYPE:") :].strip()

            elif line.startswith("DESCRIPTION:"):
                current_data["description"] = line[len("DESCRIPTION:") :].strip()

        # Save last output
        if current_output and current_data:
            outputs_list.append({current_output: current_data})

        # Store outputs in the profile
        profile["outputs"] = outputs_list

    return profiles
