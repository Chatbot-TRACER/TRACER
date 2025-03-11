import re
import os


def ensure_double_curly(text):
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


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
    IMPORTANT: Focus ONLY on user tasks and objectives.
    Do NOT let limitations drive the profile categorization – they are only to be kept in mind for refining goals.
    For example, if the functionalities are related to the CAU, the profiles should distinguish between tasks like:
      - Asking for office hours,
      - Opening a service ticket,
      - Requesting information about specific services.
    Do not mix in chatbot internal limitations (e.g., "supports only Spanish" or "handles complex questions").

    Try to cover all the important functionality groups without overlap between profiles.

    FORMAT YOUR RESPONSE AS:

    ## PROFILE: [Conversation Scenario Name]
    DESCRIPTION: [Brief description of this conversation scenario]
    FUNCTIONALITIES:
    - [functionality 1 relevant to this scenario]
    - [functionality 2 relevant to this scenario]

    ## PROFILE: [Another Conversation Scenario Name]
    DESCRIPTION: [Brief description of this scenario]
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

        description = ""
        functionalities_list = []

        description_started = False
        functionalities_started = False

        for line in lines[1:]:
            if line.startswith("DESCRIPTION:"):
                description_started = True
                description = line[len("DESCRIPTION:") :].strip()
            elif line.startswith("FUNCTIONALITIES:"):
                description_started = False
                functionalities_started = True
            elif functionalities_started and line.strip().startswith("- "):
                functionalities_list.append(line.strip()[2:])
            elif description_started:
                description += " " + line.strip()

        profiles.append(
            {
                "name": profile_name,
                "description": description,
                "functionalities": functionalities_list,
            }
        )

    # For each profile, generate user-centric goals
    for profile in profiles:
        goals_prompt = f"""
        Generate a set of coherent **user-centric** goals for this conversation scenario:

        CONVERSATION SCENARIO: {profile["name"]}
        DESCRIPTION: {profile["description"]}

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

        IMPORTANT: Your goals must be concrete tasks that a user wants to accomplish, such as opening a ticket, scheduling an appointment, or asking how to pay taxes. Do NOT include goals that reference internal chatbot characteristics or limitations
        (for instance, "ask about chatbot limitations" or "use long sentences").

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

    return profiles
