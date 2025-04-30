import re
from typing import Any

from ..constants import VARIABLE_PATTERN
from ..schemas.state import State


def ensure_double_curly(text):
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def _build_single_variable_prompt(
    profile_name: str,
    role: str,
    goals_text: str,
    variable_name: str,
    all_other_variables: list[str],
    language_instruction: str,
) -> str:
    """Creates the specific text prompt to ask the LLM about one variable.

    Args:
        profile_name: The name of the user profile we're working on.
        role: The user's role description
        goals_text: The list of goals for this profile, formatted as a string.
        variable_name: The exact name of the variable we want defined
        all_other_variables: A list of other variable names found in the same profile.
                             Helps the LLM decide if 'forward(other_var)' makes sense.
        language_instruction: A sentence telling the LLM what language to use for
                              example data values (e.g., "Generate examples in Spanish").

    Returns:
        A big string containing the full prompt to send to the LLM.
    """
    other_vars_text = (
        f"OTHER VARIABLES IN THIS PROFILE: {', '.join(sorted(all_other_variables))}" if all_other_variables else ""
    )

    prompt = f"""
    Define parameters for the variable '{variable_name}' used in a user simulator profile.

    USER PROFILE CONTEXT:
    Name: {profile_name}
    Role: {role}
    Goals (where the variable might appear):
    {goals_text}
    {other_vars_text} # Provides context for potential forward(other_var) usage.

    {language_instruction}

    Define ONLY the parameters for the variable '{variable_name}' following these guidelines:

    1. Choose ONE appropriate FUNCTION from:
        - default(): assigns ALL data in the list to the variable
        - random(): picks ONE random sample from the list
        - random(X): picks X random samples where X is LESS THAN the total number of items
        - random(rand): picks a random number of random samples
        - another(): picks different samples without repetition each time
        - another(X): picks X different samples without repetition
        - forward(): iterates through each sample one by one
        - forward(other_var): iterates and nests with other_var

    IMPORTANT FUNCTION RESTRICTIONS:
    - DO NOT nest functions (e.g., random(another(3)) is INVALID)
    - DO NOT use random(X) where X equals the total number of items (use default() instead)
    - INT() is NOT a function but a TYPE
    - Use forward() for systematic iteration through all values in order
    - Use random() for picking just one value each time (but there could be repetition)
    - Use another() when you want different values on subsequent uses

    2. Choose the most appropriate TYPE from:
        - string: for text values
        - int: for whole numbers
        - float: for decimal numbers

    3. IMPORTANT: Provide DATA in the correct format:
        - For string variables: use a list of realistic string values, the length of the list should be at least 3 items and at most 10 items, use the amount of data that makes sense for the variable
        - For int variables: ALWAYS use the min/max/step format like this:
            min: 1
            max: 10
            step: 1
        - For float variables: ALWAYS use the min/max/step format OR linspace like this:
            min: 1.0
            max: 5.0
            step: 0.5
            OR
            min: 1.0
            max: 5.0
            linspace: 5

    FORMAT YOUR RESPONSE FOR STRING VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: string
        DATA:
        - value1
        - value2
        - value3

        FORMAT YOUR RESPONSE FOR NUMERIC VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: int/float
        DATA:
        min: 1
        max: 10
        step: 1

    Respond ONLY with the FUNCTION, TYPE, and DATA for the variable '{variable_name}'.
    """
    return prompt


def _parse_single_variable_definition(
    response_content: str, expected_type: str | None = None,
) -> dict[str, Any] | None:
    """Takes the LLM's text answer for one variable and tries to turn it into a Python dictionary.

    Args:
        response_content: The raw text string that the LLM sent back.
        expected_type: (Optional) If we think we know the type (like 'string'),
                       we can pass it. It's mostly just for printing a warning
                       if the LLM gives something different. Defaults to None.

    Returns:
        A dictionary with 'function', 'type', and 'data' keys if parsing worked
        (e.g., {'function': 'random()', 'type': 'string', 'data': ['A', 'B']}).
        Returns None if the LLM's response was messed up or missing required parts.
    """
    definition = {}
    data_lines = []
    in_data_section = False
    parsed_type = None

    lines = response_content.strip().split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if line.startswith("FUNCTION:"):
            definition["function"] = line[len("FUNCTION:") :].strip()
            in_data_section = False
        elif line.startswith("TYPE:"):
            parsed_type = line[len("TYPE:") :].strip()
            definition["type"] = parsed_type
            in_data_section = False
            # If expected_type is provided, check for mismatch early
            if expected_type and parsed_type != expected_type:
                print(
                    f"Warning: LLM returned type '{parsed_type}' but expected '{expected_type}'. Will attempt to parse as '{parsed_type}'.",
                )
        elif line.startswith("DATA:"):
            in_data_section = True
            # Determine if data will be list (string) or dict (numeric)
            # Peek at the next line if available to guess format
            next_line_index = i + 1
            if next_line_index < len(lines):
                next_line = lines[next_line_index].strip()
                if ":" in next_line and not next_line.startswith("-"):
                    # Numeric (i.e. min: 1.0 max: ...)
                    definition["data"] = {}
                else:
                    # String (i.e. - value1)
                    definition["data"] = []
            else:
                # Default to list if unsure
                definition["data"] = []
        elif in_data_section:
            data_lines.append(line)

    if not definition.get("function") or not definition.get("type") or "data" not in definition:
        print(
            f"Warning: Failed to parse essential fields (FUNCTION, TYPE, DATA) from LLM response:\n{response_content}",
        )
        return None  # Essential fields missing

    # --- Post-process DATA based on TYPE ---
    data_type = definition.get("type")
    raw_data = definition["data"]  # This is either [] or {} initially

    if data_type == "string":
        # Expecting a list
        if not isinstance(raw_data, list):
            raw_data = []  # Ensure it's a list
        for item_line in data_lines:
            if item_line.startswith("- "):
                value = item_line[2:].strip().strip("'\"")
                raw_data.append(value)
        if not raw_data:
            print(f"Warning: String variable data is empty. LLM response:\n{response_content}")
            return None  # String needs data
        definition["data"] = raw_data

    elif data_type in ["int", "float"]:
        # Expecting a dict
        if not isinstance(raw_data, dict):
            raw_data = {}  # Ensure it's a dict
        for item_line in data_lines:
            if ":" in item_line:
                key, value_str = item_line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()
                try:
                    value = int(value_str) if data_type == "int" else float(value_str)
                    raw_data[key] = value
                except ValueError:
                    print(f"Warning: Could not parse numeric value for key '{key}': '{value_str}'. Skipping.")
        # Validate numeric data structure
        if not ("min" in raw_data and "max" in raw_data and ("step" in raw_data or "linspace" in raw_data)):
            print(
                f"Warning: Numeric variable data missing min/max/step or min/max/linspace. LLM response:\n{response_content}",
            )
            # Allow partial definition for potential fixing later? Or return None? Let's return None for now.
            return None
        definition["data"] = raw_data
    else:
        print(f"Warning: Unknown variable type '{data_type}'. Cannot parse data correctly.")
        return None  # Unknown type

    # Basic validation
    if not definition.get("function") or not definition.get("type") or not definition.get("data"):
        print(f"Warning: Post-parsing validation failed. Missing fields or empty data. Parsed: {definition}")
        return None

    return definition


def generate_variable_definitions(profiles, llm, supported_languages=None, max_retries=3):
    """Generate the {{variable}} definitions.

    Goes through user profiles, finds variables like {{this}} in their goals,
    and asks the LLM to define them (type, function, data).

    Args:
        profiles: A list where each item is a dictionary representing a user profile.
                  Needs to have a 'goals' key containing a list of strings.
        llm: The language model object (like ChatOpenAI) used to get definitions.
        supported_languages: (Optional) A list of languages the chatbot knows.
                             The first one is used for examples. Defaults to None.
        max_retries: (Optional) How many times to try asking the LLM again if
                     parsing its response fails for a variable. Defaults to 3.

    Returns:
        The same list of profiles that was passed in, but now the 'goals' list
        inside each profile dictionary also contains the variable definition
        dictionaries (e.g., {'variable_name': {'function': ..., 'type': ..., 'data': ...}}).
    """
    primary_language = ""
    language_instruction = ""
    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Generate examples/values in {primary_language} where appropriate."

    # Process each profile
    for profile in profiles:
        # Extract all unique variables from the goals
        all_variables: set[str] = set()
        goals_list = profile.get("goals", [])
        if not isinstance(goals_list, list):  # Ensure goals is a list
            print(
                f"Warning: Profile '{profile.get('name', 'Unnamed')}' has invalid 'goals' format. Skipping variable definition.",
            )
            continue

        for goal in goals_list:
            if isinstance(goal, str):
                variables_in_goal = VARIABLE_PATTERN.findall(goal)
                all_variables.update(variables_in_goal)
            # Ignore non-string goals (like existing variable definitions if run multiple times)

        if not all_variables:
            print(f"Info: No variables found in goals for profile '{profile.get('name', 'Unnamed')}'.")
            continue

        print(f"\n--- Defining variables for profile: {profile.get('name', 'Unnamed')} ---")
        print(f"   Found variables: {', '.join(sorted(all_variables))}")

        goals_text = ""
        for goal in goals_list:
            if isinstance(goal, str):
                goals_text += f"- {goal}\n"  # Only include string goals in the context

        # Define each variable individually
        for variable_name in sorted(all_variables):  # Sort for consistent order
            print(f"   Defining variable: '{variable_name}'...")
            other_variables = list(all_variables - {variable_name})
            parsed_def = None

            for attempt in range(max_retries):
                prompt = _build_single_variable_prompt(
                    profile.get("name", "Unnamed Profile"),
                    profile.get("role", "Unknown Role"),
                    goals_text,
                    variable_name,
                    other_variables,
                    language_instruction,
                )
                try:
                    response = llm.invoke(prompt)
                    response_content = response.content

                    parsed_def = _parse_single_variable_definition(response_content)

                    if parsed_def:
                        print(f"      Successfully parsed definition for '{variable_name}'.")
                        break  # Success
                    print(
                        f"      Warning: Failed to parse definition for '{variable_name}' on attempt {attempt + 1}. Retrying...",
                    )

                except Exception as e:
                    print(
                        f"      Error during LLM call or parsing for '{variable_name}' on attempt {attempt + 1}: {e}. Retrying...",
                    )
                    parsed_def = None  # Ensure reset on error

            # Store the result (or handle failure)
            if parsed_def:
                # Add the definition as a dictionary under the variable name key
                # Check if a definition already exists (e.g., from a previous run or manual entry)
                existing_def_index = -1
                for i, goal_item in enumerate(goals_list):
                    if isinstance(goal_item, dict) and variable_name in goal_item:
                        existing_def_index = i
                        break

                if existing_def_index != -1:
                    print(f"      Updating existing definition for '{variable_name}'.")
                    goals_list[existing_def_index] = {variable_name: parsed_def}
                else:
                    print(f"      Adding new definition for '{variable_name}'.")
                    goals_list.append({variable_name: parsed_def})  # Append new definition dict

            else:
                print(
                    f"      ERROR: Failed to generate valid definition for variable '{variable_name}' after {max_retries} attempts.",
                )

        # Update the profile's goals list with the added/updated definitions
        profile["goals"] = goals_list

    return profiles


def generate_context(profiles, functionalities, llm, supported_languages=None):
    """Generate additional context for the user simulator as multiple short entries."""
    # Work in the detected primary language
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Write the context in {primary_language}."

    for profile in profiles:
        # Replace variables in goals with general terms for context generation
        sanitized_goals = []
        for goal in profile.get("goals", []):
            # Replace {{variable}} with general terms
            if isinstance(goal, str):
                sanitized_goal = re.sub(VARIABLE_PATTERN, "[specific details]", goal)
                sanitized_goals.append(sanitized_goal)

        context_prompt = f"""
        Create 2-3 SHORT context points for a user simulator interacting with a chatbot.
        Each point should be a separate piece of background information or context that helps the simulator.

        CONVERSATION SCENARIO: {profile["name"]}
        CURRENT USER ROLE: {profile["role"]}
        USER GOALS (generalized):
        {", ".join(sanitized_goals)}

        {language_instruction}

        GUIDELINES:
        1. Write 2-3 SEPARATE short context points, each 1-2 sentences only
        2. Each point should focus on ONE aspect (background info, knowledge, motivation)
        3. NEVER include variables like {{date}} or {{amount}} - use specific examples instead
        4. Keep each point brief and focused
        5. Make each point distinctly different from the others

        Examples of GOOD context points:
        - "You tried calling the office yesterday but no one answered."
        - "You need to finish this task before your meeting at 3pm."
        - "Your colleague mentioned this service was very reliable."

        FORMAT YOUR RESPONSE WITH ONE CONTEXT POINT PER LINE:
        - First context point
        - Second context point
        - Third context point (optional)

        IMPORTANT: Each line should start with "- " and be a standalone piece of context.
        """

        context_response = llm.invoke(context_prompt)
        context_content = context_response.content.strip()

        # Process context into separate entries
        context_entries = []
        for line in context_content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                entry = line[2:].strip()
                if entry:
                    context_entries.append(entry)

        profile["context"] = context_entries

    return profiles


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


def generate_user_profiles_and_goals(
    functionalities,
    limitations,
    llm,
    workflow_structure=None,
    conversation_history=None,
    output_dir="profiles",
    supported_languages=None,
    chatbot_type="unknown",
):
    """Generate user profiles and goals based on functionalities and limitations.

    Group functionalities into logical user profiles and generate coherent goal sets
    for individual conversations

    Args:
        functionalities: List of discovered functionality descriptions
        limitations: List of discovered limitations
        llm: Language model to use for generation
        workflow_structure: Optional structure representing workflow relationships between functionalities
        conversation_history: Optional historical conversation data
        output_dir: Directory to save output profiles
        supported_languages: List of supported languages
        chatbot_type: Type of chatbot ("transactional", "informational", or "unknown")

    Returns:
        List of profile dictionaries with goals and variable definitions
    """
    # Work in the given language with stronger instructions
    primary_language = ""
    language_instruction_grouping = ""
    language_instruction_goals = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        # More specific instruction with examples that will help the model follow format
        language_instruction_grouping = f"""
LANGUAGE REQUIREMENT:
- Write ALL profile names, descriptions, and functionalities in {primary_language}
- KEEP ONLY the formatting markers (##, PROFILE:, DESCRIPTION:, FUNCTIONALITIES:) in English
- Example if the primary language was Spanish:
  ## PROFILE: [Nombre del escenario en Spanish]
  DESCRIPTION: [Descripción en Spanish]
  FUNCTIONALITIES:
  - [Funcionalidad en Spanish]
"""

        language_instruction_goals = f"""
LANGUAGE REQUIREMENT:
- Write ALL goals in {primary_language}
- KEEP ONLY the formatting markers (GOALS:) in English
- Keep variables in {{variable}} format
- Example in Spanish:
  GOALS:
  - "Primer objetivo en Spanish con {{variable}}"
"""

    # Prepare a condensed version of conversation history if available
    conversation_context = ""
    if conversation_history:
        conversation_context = "Here are some example conversations with the chatbot:\n\n"
        for i, session in enumerate(conversation_history, 1):
            conversation_context += f"--- SESSION {i} ---\n"
            for turn in session:
                if turn["role"] == "assistant":  # Explorer
                    conversation_context += f"Human: {turn['content']}\n"
                elif turn["role"] == "user":  # Chatbot's response
                    conversation_context += f"Chatbot: {turn['content']}\n"
            conversation_context += "\n"

    # Prepare workflow information if available
    workflow_context = ""
    if workflow_structure:
        workflow_context = "WORKFLOW INFORMATION (how functionalities connect):\n"

        # Process the workflow structure to extract relationships
        for node in workflow_structure:
            if isinstance(node, dict):
                node_name = node.get("name", "")
                node_children = node.get("children", [])

                if node_name and node_children:
                    child_names = [
                        child.get("name", "") for child in node_children if isinstance(child, dict) and "name" in child
                    ]
                    if child_names:
                        workflow_context += f"- {node_name} can lead to: {', '.join(child_names)}\n"
                elif node_name:
                    workflow_context += f"- {node_name} (standalone functionality)\n"

    # Include chatbot type information
    chatbot_type_context = f"CHATBOT TYPE: {chatbot_type.upper()}\n"

    # Ask the LLM to identify distinct conversation scenarios
    # Calculate an appropriate number of profiles based on functionality count
    num_functionalities = len(functionalities)
    min_profiles = 3
    max_profiles = 10

    suggested_profiles = max(min_profiles, min(max_profiles, num_functionalities))

    # Update the grouping prompt to use the calculated number
    grouping_prompt = f"""
    Based on these chatbot functionalities:
    {", ".join(functionalities)}

    {conversation_context}
    {workflow_context}
    {chatbot_type_context}

    {language_instruction_grouping}

    Create {suggested_profiles} distinct user profiles, where each profile represents ONE specific conversation scenario.

    EXTREMELY IMPORTANT RESTRICTIONS:
    1. Create ONLY profiles for realistic end users with PRACTICAL GOALS
    2. NEVER create profiles about users asking about chatbot capabilities or limitations
    3. NEVER create profiles where the user is trying to test or evaluate the chatbot
    4. Focus ONLY on real-world user tasks and objectives
    5. The profiles should be genuine use cases, not meta-conversations about the chatbot itself

    For example, if the functionalities are related to the CAU, the profiles should distinguish between tasks like:
      - Asking for office hours,
      - Opening a service ticket,
      - Requesting information about specific services.
    Do not mix in chatbot internal limitations (e.g., "supports only Spanish" or "handles complex questions").

    Try to cover all the important functionality groups without overlap between profiles.

    FORMAT YOUR RESPONSE AS:

    ## PROFILE: [Conversation Scenario Name]
    ROLE: [Write a prompt for the user simulator, e.g. "you have to act as a user ordering a pizza to a pizza shop."]
    FUNCTIONALITIES:
    - [functionality 1 relevant to this scenario]
    - [functionality 2 relevant to this scenario]

    ## PROFILE: [Another Conversation Scenario Name]
    ROLE: [Write a prompt for the user simulator, e.g. "you have to act as a user ordering a pizza to a pizza shop."]
    FUNCTIONALITIES:
    - [functionality 3 relevant to this scenario]
    - [functionality 4 relevant to this scenario]

    ... and so on
    """

    # Get scenario groupings from the LLM
    profiles_response = llm.invoke(grouping_prompt)
    profiles_content = profiles_response.content

    # Parse the profiles
    profile_sections = profiles_content.split("## PROFILE:")

    if not profile_sections[0].strip():
        profile_sections = profile_sections[1:]

    profiles = []

    for section in profile_sections:
        lines = section.strip().split("\n")
        profile_name = lines[0].strip()

        role = ""
        functionalities_list = []

        role_started = False
        func_started = False

        for line in lines[1:]:
            if line.startswith("ROLE:"):
                role_started = True
                role = line[len("ROLE:") :].strip()
                func_started = False
            elif line.startswith("FUNCTIONALITIES:"):
                role_started = False
                func_started = True
            elif func_started and line.strip().startswith("- "):
                functionalities_list.append(line.strip()[2:].strip())
            elif role_started:
                role += " " + line.strip()

        profiles.append(
            {
                "name": profile_name,
                "role": role.strip(),
                "functionalities": functionalities_list,
            },
        )

    # For each profile, generate user-centric goals
    for profile in profiles:
        goals_prompt = f"""
        Generate a set of coherent **user-centric** goals for this conversation scenario:

        CONVERSATION SCENARIO: {profile["name"]}
        ROLE: {profile["role"]}

        {chatbot_type_context}

        RELEVANT FUNCTIONALITIES:
        {", ".join(profile["functionalities"])}

        {workflow_context}

        LIMITATIONS (keep in mind only; do NOT let these drive the goals):
        {", ".join(limitations)}

        {conversation_context}

        {language_instruction_goals}

        ABOUT VARIABLES:
        - Only use {{variable}} where the user might provide different values each time (e.g. {{date}}, {{amount}}, {{reference_number}})
        - These are purely placeholders for possible user input. For example, {{employee_id}} does not mean we must always request an ID; it's just a potential input that could vary.
        - Do NOT put fixed names like "IT Department" or organization names inside {{ }} (they are not interchangeable).
        - Variables must be legitimate parameters the user could change (e.g., different dates, amounts, or IDs).

        EXTREMELY IMPORTANT RESTRICTIONS:
        1. NEVER create goals about asking for chatbot limitations or capabilities
        2. NEVER create goals about testing the chatbot's understanding or knowledge
        3. NEVER include meta-goals like "find out what the chatbot can do"
        4. Goals MUST be about actual tasks a real user would want to accomplish
        5. Focus on practical, realistic user tasks ONLY

        Create 2-4 goals that focus strictly on what the user intends to achieve with the chatbot.
        Avoid vague or indirect objectives like "understand the chatbot's capabilities" or "test the system's knowledge."

        IMPORTANT:
        - If the chatbot is TRANSACTIONAL, goals should follow a natural workflow progression. Create goals that represent steps in completing a process or transaction.
        - If the chatbot is INFORMATIONAL, goals can be more independent questions, but should still be related to the same general topic.
        - If workflow information is provided, create goals that follow natural conversation flows discovered during exploration.

        Examples for TRANSACTIONAL chatbots:

        Example 1 (IT Support):
        - "Report a technical issue with my {{device_type}}"
        - "Provide additional details about the problem"
        - "Request an estimated resolution time"
        - "Ask for a ticket confirmation number"

        Example 2 (Appointment Scheduling):
        - "Schedule an appointment for {{service_type}}"
        - "Select a preferred date from {{available_dates}}"
        - "Confirm the appointment details"
        - "Request a reminder option"

        Examples for INFORMATIONAL chatbots:

        Example 1 (University Information):
        - "Ask about admission requirements for {{program_name}}"
        - "Request information about application deadlines"
        - "Inquire about scholarship opportunities"

        Example 2 (Government Services):
        - "Ask about the process for renewing a {{document_type}}"
        - "Inquire about required documentation"
        - "Find out about processing times"

        FORMAT YOUR RESPONSE AS:

        GOALS:
        - "first user-centric goal with {{variable}} if needed"
        - "second related goal"
        - "third goal that follows naturally"

        DO NOT include any definitions for variables - just use {{varname}} placeholders.
        Make sure all goals fit naturally in ONE conversation with the chatbot, and remain strictly focused on user tasks.
        """

        goals_response = llm.invoke(goals_prompt)
        goals_content = goals_response.content

        goals = []
        if "GOALS:" in goals_content:
            goals_section = goals_content.split("GOALS:")[1].strip()
            for line in goals_section.split("\n"):
                if line.strip().startswith("- "):
                    goal = line.strip()[2:].strip().strip("\"'")
                    if goal:
                        goal = ensure_double_curly(goal)
                        goals.append(goal)

        profile["goals"] = goals

    # Generate values for the variables
    profiles = generate_variable_definitions(profiles, llm, supported_languages)

    # Generate context
    profiles = generate_context(profiles, functionalities, llm, supported_languages)

    # Generate output fields
    profiles = generate_outputs(profiles, functionalities, llm, supported_languages)
    return profiles


def goal_generator_node(state: State, llm) -> dict[str, Any]:
    """Node that generates user profiles and conversation goals based on findings.

    Args:
        state (State): The current graph state.
        llm: The language model instance.

    Returns:
        Dict[str, Any]: Dictionary with updated 'conversation_goals'.
    """
    # This node is part of the profile generation graph.
    # It expects 'discovered_functionalities' (structured) and 'chatbot_type' from the previous graph.

    if not state.get("discovered_functionalities"):
        print("\n--- Skipping goal generation: No structured functionalities found. ---")
        return {"conversation_goals": []}

    print("\n--- Generating conversation goals from structured data ---")

    # Functionalities are now dicts (structured from previous node)
    structured_root_dicts: list[dict[str, Any]] = state["discovered_functionalities"]

    # Get workflow structure (which is the structured functionalities itself)
    workflow_structure = structured_root_dicts  # Use the structured data directly

    # Get chatbot type from state
    chatbot_type = state.get("chatbot_type", "unknown")
    print(f"   Chatbot type for goal generation: {chatbot_type}")

    # Helper to get all descriptions from the structure
    def get_all_descriptions(nodes: list[dict[str, Any]]) -> list[str]:
        descriptions = []
        for node in nodes:
            if node.get("description"):
                descriptions.append(node["description"])
            if node.get("children"):
                child_descriptions = get_all_descriptions(node["children"])
                descriptions.extend(child_descriptions)
        return descriptions

    functionality_descriptions = get_all_descriptions(structured_root_dicts)

    if not functionality_descriptions:
        print("   Warning: No descriptions found in structured functionalities.")
        return {"conversation_goals": []}

    print(f" -> Preparing {len(functionality_descriptions)} descriptions (from structure) for goal generation.")

    try:
        # Call the goal generation function
        profiles_with_goals = generate_user_profiles_and_goals(
            functionality_descriptions,
            state.get("discovered_limitations", []),  # Limitations might not be populated
            llm,
            workflow_structure=workflow_structure,
            conversation_history=state.get("conversation_history", []),
            supported_languages=state.get("supported_languages", []),
            chatbot_type=chatbot_type,
        )
        print(f" -> Generated {len(profiles_with_goals)} profiles with goals.")
        # Update state with goals
        return {"conversation_goals": profiles_with_goals}

    except Exception as e:
        print(f"Error during goal generation: {e}")
        return {"conversation_goals": []}  # Return empty list on error
