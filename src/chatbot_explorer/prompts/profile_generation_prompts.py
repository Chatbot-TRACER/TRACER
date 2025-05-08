"""Prompts for grouping functionalities and generating profile goals."""

from typing import Any, TypedDict

# --- Data Structures for Prompt Arguments ---


class ProfileGroupingContext(TypedDict):
    """Contextual information for grouping functionalities into profiles.

    Args:
        conversation_context: String describing the overall conversation history/context.
        workflow_context: String describing discovered conversation workflows.
        chatbot_type_context: String describing the type of chatbot (transactional/informational).
        language_instruction_grouping: Language-specific instructions for the grouping prompt.
    """

    conversation_context: str
    workflow_context: str
    chatbot_type_context: str
    language_instruction_grouping: str


class ProfileGoalContext(TypedDict):
    """Contextual information for generating goals for a specific profile.

    Args:
        chatbot_type_context: String describing the type of chatbot.
        workflow_context: String describing discovered conversation workflows.
        limitations: List of known chatbot limitations.
        conversation_context: String describing the overall conversation history/context.
        language_instruction_goals: Language-specific instructions for the goal generation prompt.
    """

    chatbot_type_context: str
    workflow_context: str
    limitations: list[str]
    conversation_context: str
    language_instruction_goals: str


# --- Language Instruction Functions ---


def get_language_instruction_grouping(primary_language: str) -> str:
    """Generate the language instruction for profile grouping."""
    if not primary_language:
        return ""
    return f"""
LANGUAGE REQUIREMENT:
- Write ALL profile names, roles, and functionalities in {primary_language}
- KEEP ONLY the formatting markers (##, PROFILE:, ROLE:, FUNCTIONALITIES:) in English
- Example if the primary language was Spanish:
  ## PROFILE: [Nombre del escenario en Spanish]
  ROLE: [Rol del usuario en Spanish]
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


# --- Prompt Generation Functions ---


def get_profile_grouping_prompt(
    functionalities: list[str],
    context: ProfileGroupingContext,
    suggested_profiles: int,
) -> str:
    """Generates a prompt to group functionalities into logical user profiles."""
    functionality_list_str = "\n".join([f"- {f}" for f in functionalities])
    num_functionalities = len(functionalities)

    # Construct context section only if parts are available
    context_str = "\n--- CONTEXTUAL INFORMATION ---\n"
    has_context = False
    if context.get("chatbot_type_context"):
        context_str += context["chatbot_type_context"]
        has_context = True
    if context.get("workflow_context"):
        context_str += context["workflow_context"]
        has_context = True
    if context.get("conversation_context"):
        max_conv_len = 2000
        conv_snippet = context["conversation_context"]
        if len(conv_snippet) > max_conv_len:
            conv_snippet = conv_snippet[:max_conv_len] + "\n... (conversation truncated)"
        context_str += "\nSAMPLE CONVERSATIONS:\n" + conv_snippet + "\n"
        has_context = True

    if not has_context:
        context_str = ""
    else:
        context_str += "--- END CONTEXTUAL INFORMATION ---\n\n"

    return f"""
You are a User Profile Designer for chatbot testing. Your task is to group the following chatbot functionalities into logical user profiles. The goal is to create distinct user personas/roles, each focused on achieving a related set of goals using a subset of the chatbot's capabilities, ensuring all functionalities are covered across the profiles.

EXTREMELY IMPORTANT RESTRICTIONS:

    Create ONLY profiles for realistic end users with PRACTICAL GOALS

    NEVER create profiles about users asking about chatbot capabilities or limitations

    NEVER create profiles where the user is trying to test or evaluate the chatbot

    Focus ONLY on real-world user tasks and objectives

    The profiles should be genuine use cases, not meta-conversations about the chatbot itself

AVAILABLE CHATBOT FUNCTIONALITIES ({num_functionalities} total):
{functionality_list_str}
{context_str}
**Instructions for Grouping:**

1.  **Analyze Relationships:** Examine the list of functionalities. Identify functionalities that are related or often used together in a single user journey or to achieve a larger goal. Use the WORKFLOW INFORMATION (if provided) to see which functionalities often follow others.
2.  **Define Coherent Roles/Personas:** Create distinct user roles or personas. Each role should represent a plausible type of user with a specific set of needs or tasks they want to accomplish using the chatbot (e.g., "New Customer Inquiry," "Existing User Order Tracking," "Product Configurator," "Information Seeker about Topic X").
3.  **Group Functionalities Logically:** Assign each functionality to the *most relevant* user profile/role. A profile should contain a set of functionalities that make sense for that specific user persona to use together.
    *   Group steps that form a sequence (e.g., select item, add to cart, checkout).
    *   Group informational functions related to a specific topic (e.g., all functions related to pricing, all related to account settings).
    *   Group functions related to a specific high-level task (e.g., placing a new order, checking status, getting support).
4.  **Ensure Coverage:** Distribute ALL functionalities across the profiles. Each functionality should ideally belong to at least one profile. Avoid leaving functionalities unassigned.
5.  **Balance Profile Size (Guideline):** Aim for roughly {suggested_profiles} distinct profiles, but prioritize logical coherence over strictly hitting this number. It's better to have fewer, well-defined profiles or slightly more if needed for logical separation. Avoid profiles with only one trivial functionality if it can logically fit elsewhere. Don't create overly large profiles containing unrelated functions.
6.  **Describe the Role:** For each profile, provide a concise description of the user's role or primary objective.
7.  **Output Format:** Structure your response exactly as follows, with each profile starting with `## PROFILE:`, followed by the profile name, role, and functionalities list:

## PROFILE: [Profile Name (e.g., New Order Placer)]
ROLE: [Description of the user role/objective for this profile]
FUNCTIONALITIES:
- [Functionality description assigned to this profile]
- [Another functionality description assigned to this profile]
...

## PROFILE: [Another Profile Name (e.g., Account Manager)]
ROLE: [Description of this role]
FUNCTIONALITIES:
- [Functionality description]
...

{context["language_instruction_grouping"]}
Make sure the profile names and roles are descriptive and distinct. Ensure all functionalities from the input list are assigned to a profile.
"""  # noqa: S608


def get_profile_goals_prompt(
    profile: dict[str, Any],
    context: ProfileGoalContext,
) -> str:
    """Returns the prompt for generating user-centric goals for a profile.

    Args:
        profile: The specific profile dictionary (containing name, role, functionalities).
        context: A dictionary containing various contextual strings for the prompt.

    Returns:
        A formatted string representing the LLM prompt.
    """
    return f"""
Generate a set of coherent **user-centric** goals for this conversation scenario:

CONVERSATION SCENARIO: {profile["name"]}
ROLE: {profile["role"]}

{context["chatbot_type_context"]}

RELEVANT FUNCTIONALITIES:
{", ".join(profile["functionalities"])}

{context["workflow_context"]}

LIMITATIONS (keep in mind only; do NOT let these drive the goals):
{", ".join(context["limitations"])}

{context["conversation_context"]}

{context["language_instruction_goals"]}

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
"""  # noqa: S608
