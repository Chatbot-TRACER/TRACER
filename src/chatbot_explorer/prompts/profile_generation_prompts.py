from typing import Any


def get_language_instruction_grouping(primary_language: str) -> str:
    """Generate the language instruction for profile grouping."""
    if not primary_language:
        return ""
    return f"""
LANGUAGE REQUIREMENT:
- Write ALL profile names, descriptions, and functionalities in {primary_language}
- KEEP ONLY the formatting markers (##, PROFILE:, DESCRIPTION:, FUNCTIONALITIES:) in English
- Example if the primary language was Spanish:
  ## PROFILE: [Nombre del escenario en Spanish]
  DESCRIPTION: [Descripción en Spanish]
  FUNCTIONALITIES:
  - [Funcionalidad en Spanish]
"""


def get_language_instruction_goals(primary_language: str) -> str:
    """Generate the language instruction for goal generation."""
    if not primary_language:
        return ""
    return f"""
LANGUAGE REQUIREMENT:
- Write ALL goals in {primary_language}
- KEEP ONLY the formatting markers (GOALS:) in English
- Keep variables in {{variable}} format
- Example in Spanish:
  GOALS:
  - "Primer objetivo en Spanish con {{variable}}"
"""


def get_profile_grouping_prompt(
    functionalities: list[str],
    conversation_context: str,
    workflow_context: str,
    chatbot_type_context: str,
    language_instruction_grouping: str,
    suggested_profiles: int,
) -> str:
    """Returns the prompt for grouping functionalities into profile scenarios."""
    return f"""
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


def get_profile_goals_prompt(
    profile: dict[str, Any],
    chatbot_type_context: str,
    workflow_context: str,
    limitations: list[str],
    conversation_context: str,
    language_instruction_goals: str,
) -> str:
    """Returns the prompt for generating user-centric goals for a profile."""
    return f"""
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
