import re
import os

VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def ensure_double_curly(text):
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def generate_variable_definitions(profiles, llm, supported_languages=None):
    """
    Extract variables from goals and generate appropriate definitions
    using an LLM to determine type, function and data values.
    """

    # Work in the given language for consistency
    primary_language = ""
    language_instruction = ""

    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = f"Follow these language instructions: Generate examples in {primary_language} where appropriate."

    # Process each profile to find variables and generate definitions
    for profile in profiles:
        # Extract all variables from the goals
        all_variables = set()
        for goal in profile.get("goals", []):
            variables = VARIABLE_PATTERN.findall(goal)
            all_variables.update(variables)

        if not all_variables:
            continue

        # Generate definitions for all discovered variables
        vars_prompt = f"""
        I need to define variable parameters for a user simulator that interacts with a chatbot.

        USER PROFILE: {profile["name"]}
        ROLE: {profile["role"]}

        GOALS:
        {chr(10).join(f"- {goal}" for goal in profile["goals"])}

        {language_instruction}

        For each variable I listed below, provide a definition following these guidelines:

        1. Choose the most appropriate FUNCTION from:
           - default(): assigns all data in the list to the variable
           - random(): picks one random sample from the list
           - random(X): picks X random samples from the list
           - random(rand): picks a random number of random samples
           - another(): picks different samples without repetition
           - another(X): picks X different samples without repetition
           - forward(): iterates through each sample one by one
           - forward(other_var): iterates and nests with other_var

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

        VARIABLES TO DEFINE:
        {", ".join(sorted(all_variables))}

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

        PROVIDE DEFINITIONS FOR ALL VARIABLES, one after another.
        Remember: ALL numeric variables (int/float) MUST use the min/max/step format, NOT a list of values.
        """

        # Get definitions from the LLM
        definitions_response = llm.invoke(vars_prompt)
        definitions_content = definitions_response.content

        # Parse the definitions
        current_var = None
        current_def = {}
        stage = None

        for line in definitions_content.split("\n"):
            line = line.strip()

            if not line:
                continue

            if line.startswith("VARIABLE:"):
                # Save previous variable if exists
                if current_var and current_def:
                    profile[current_var] = current_def

                # Start new variable
                current_var = line[len("VARIABLE:") :].strip()
                current_def = {}
                stage = None

            elif line.startswith("FUNCTION:"):
                current_def["function"] = line[len("FUNCTION:") :].strip()

            elif line.startswith("TYPE:"):
                current_def["type"] = line[len("TYPE:") :].strip()

            elif line.startswith("DATA:"):
                current_def["data"] = []
                stage = "data"

            elif stage == "data":
                if line.startswith("- "):
                    # List item
                    value = line[2:].strip()
                    # Convert int/float strings to appropriate type
                    if current_def.get("type") == "int" and value.isdigit():
                        value = int(value)
                    elif current_def.get("type") == "float":
                        value = float(value)
                    current_def["data"].append(value)
                elif ":" in line:
                    # min/max/step format
                    if (
                        isinstance(current_def["data"], list)
                        and not current_def["data"]
                    ):
                        current_def["data"] = {}
                    key, value = line.split(":", 1)
                    key = key.strip()
                    try:
                        value = (
                            int(value.strip())
                            if current_def.get("type") == "int"
                            else float(value.strip())
                        )
                        if isinstance(current_def["data"], dict):
                            current_def["data"][key] = value
                    except ValueError:
                        pass

        # Save the last variable
        if current_var and current_def:
            profile[current_var] = current_def

    return profiles


def generate_user_profiles_and_goals(
    functionalities,
    limitations,
    llm,
    conversation_history=None,
    output_dir="profiles",
    supported_languages=None,
):
    """
    Group functionalities into logical user profiles and generate coherent goal sets
    for individual conversations
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
        conversation_context = (
            "Here are some example conversations with the chatbot:\n\n"
        )
        for i, session in enumerate(conversation_history, 1):
            conversation_context += f"--- SESSION {i} ---\n"
            for turn in session:
                if turn["role"] == "assistant":  # Explorer
                    conversation_context += f"Human: {turn['content']}\n"
                elif turn["role"] == "user":  # Chatbot's response
                    conversation_context += f"Chatbot: {turn['content']}\n"
            conversation_context += "\n"

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
            }
        )

    # For each profile, generate user-centric goals
    for profile in profiles:
        goals_prompt = f"""
        Generate a set of coherent **user-centric** goals for this conversation scenario:

        CONVERSATION SCENARIO: {profile["name"]}
        ROLE: {profile["role"]}

        RELEVANT FUNCTIONALITIES:
        {", ".join(profile["functionalities"])}

        LIMITATIONS (keep in mind only; do NOT let these drive the goals):
        {", ".join(limitations)}

        {conversation_context}

        {language_instruction_goals}

        ABOUT VARIABLES:
        - Only use {{variable}} where the user might provide different values each time (e.g. {{date}}, {{amount}}, {{phone_number}}, {{dog_breed}})
        - These are purely placeholders for possible user input. For example, {{phone_number}} does not mean we must always request a phone number; it’s just a potential input that could vary.
        - Do NOT put fixed names like "Centro de Atención a Usuarios" or "CAU" inside {{ }} (they are not interchangeable).
        - Variables must be legitimate parameters the user could change (e.g., different dates, amounts, or IDs).

        EXTREMELY IMPORTANT RESTRICTIONS:
        1. NEVER create goals about asking for chatbot limitations or capabilities
        2. NEVER create goals about testing the chatbot's understanding or knowledge
        3. NEVER include meta-goals like "find out what the chatbot can do"
        4. Goals MUST be about actual tasks a real user would want to accomplish
        5. Focus on practical, realistic user tasks ONLY

        Create 2-4 goals that focus strictly on what the user intends to achieve with the chatbot.
        Avoid vague or indirect objectives like "consultar las limitaciones del chatbot" or "solicitar ejemplos sobre el CAU."

        Examples of good goal sets:

        Example 1 (Food ordering):
        - "Order a {{size}} pizza with {{toppings}}"
        - "Add {{quantity}} {{drink}} to my order"
        - "Ask about delivery time"
        - "Get my order total and confirmation number"

        Example 2 (Municipal services):
        - "Ask about property tax"
        - "Find out how to pay it"

        Example 3 (City registration):
        - "Ask how to register as a resident"
        - "Find out what documents are needed"
        - "Ask if registration can be done online"

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

    return profiles
