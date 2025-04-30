import re

from chatbot_explorer.generation.context_generation import generate_context
from chatbot_explorer.generation.output_generation import generate_outputs
from chatbot_explorer.generation.variable_definition import generate_variable_definitions


def ensure_double_curly(text):
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def generate_profile_content(
    functionalities,
    limitations,
    llm,
    workflow_structure=None,
    conversation_history=None,
    output_dir="profiles",
    supported_languages=None,
    chatbot_type="unknown",
):
    """Generates the content of what the profile contains.

    Orchestrates the generation of:
    - Profile scenarios (name, role) based on functionality grouping.
    - User-centric goals for each scenario.
    - Variable definitions ({{variable}}) within goals.
    - Context points for the simulator.
    - Expected output fields to extract from chatbot responses.

    Args:
        functionalities: List of discovered functionality descriptions.
        limitations: List of discovered limitations.
        llm: Language model to use for generation.
        workflow_structure: Optional structure representing workflow relationships.
        conversation_history: Optional historical conversation data.
        supported_languages: List of supported languages.
        chatbot_type: Type of chatbot ("transactional", "informational", or "unknown").

    Returns:
        List of dictionaries, where each dictionary represents a complete profile's content.
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
    profiles = generate_context(profiles, llm, supported_languages)

    # Generate output fields
    return generate_outputs(profiles, functionalities, llm, supported_languages)
