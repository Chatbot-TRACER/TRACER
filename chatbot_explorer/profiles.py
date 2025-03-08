import os


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
    # First, create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    grouping_prompt = f"""
    Based on these chatbot functionalities:
    {", ".join(functionalities)}

    And these limitations:
    {", ".join(limitations)}

    {conversation_context}

    {language_instruction_grouping}

    Create 3-5 distinct user profiles, where each profile represents ONE specific conversation scenario.

    IMPORTANT: Each profile should contain goals that make sense to accomplish in a SINGLE conversation.
    For example, "ordering food and checking delivery time" is ONE conversation scenario, while
    "filing taxes and asking about community events" would be TWO separate scenarios.

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

    # Skip the first element if it's empty
    if not profile_sections[0].strip():
        profile_sections = profile_sections[1:]

    profiles = []

    # Process each profile section
    for section in profile_sections:
        lines = section.strip().split("\n")
        profile_name = lines[0].strip()

        # Extract description
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

    # For each profile, generate appropriate goals for a single conversation
    for profile in profiles:
        goals_prompt = f"""
        Generate a set of coherent goals for this conversation scenario:

        CONVERSATION SCENARIO: {profile["name"]}
        DESCRIPTION: {profile["description"]}

        RELEVANT FUNCTIONALITIES:
        {", ".join(profile["functionalities"])}

        LIMITATIONS:
        {", ".join(limitations)}

        {conversation_context}

        {language_instruction_goals}

        Create 2-4 goals that form a NATURAL CONVERSATION FLOW within this single scenario.
        All goals should logically connect as part of ONE user's interaction.

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
        - "first goal with {{variable}} if needed"
        - "second related goal"
        - "third goal that follows naturally"

        DO NOT include variable definitions - just use {{varname}} placeholders.
        Make sure all goals fit naturally in ONE conversation with the chatbot.
        """

        # Get goals for this profile
        goals_response = llm.invoke(goals_prompt)
        goals_content = goals_response.content

        # Extract just the goals list
        goals = []
        if "GOALS:" in goals_content:
            goals_section = goals_content.split("GOALS:")[1].strip()
            for line in goals_section.split("\n"):
                if line.strip().startswith("- "):
                    # Clean up the goal text (remove quotes and extra spaces)
                    goal = line.strip()[2:].strip().strip("\"'")
                    if goal:  # Only add non-empty goals
                        goals.append(goal)

        profile["goals"] = goals

        # Save to a simple text file
        filename = f"{profile['name'].lower().replace(' ', '_').replace(',', '').replace('&', 'and')}_profile.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as file:
            file.write(f"# User Profile: {profile['name']}\n")
            file.write(f"# Description: {profile['description']}\n\n")
            file.write("# Relevant Functionalities:\n")
            for func in profile["functionalities"]:
                file.write(f"# - {func}\n")

            file.write("\n# Goals for a single conversation:\n")
            for goal in profile["goals"]:
                file.write(f"- {goal}\n")

        profile["file_path"] = filepath

    return profiles
