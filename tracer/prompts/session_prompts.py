"""Prompts for managing chatbot exploration sessions."""

from tracer.schemas.functionality_node_model import FunctionalityNode

MAX_CONSECUTIVE_FAILURES = 2


def get_session_focus(current_node: FunctionalityNode | None) -> str:
    """Determine the focus string for the exploration session."""
    if current_node:
        # Focus on the specific node
        focus = f"Focus on actively using and exploring the '{current_node.name}' functionality ({current_node.description}). If it requires input, try providing plausible values. If it offers choices, select one to proceed."
        if current_node.parameters:
            param_names = [p.name for p in current_node.parameters]
            focus += f" Attempt to provide values for parameters like: {', '.join(param_names)}."
        return focus
    # General exploration focus
    return "Explore the chatbot's main capabilities. Ask what it can do or what topics it covers. If it offers options or asks questions requiring a choice, TRY to provide an answer or make a selection to see where it leads."


def get_language_instruction(supported_languages: list[str] | None, primary_language: str) -> str:
    """Generate the language instruction string for the system prompt."""
    if supported_languages:
        language_str = ", ".join(supported_languages)
        # Use primary_language in the instruction for clarity
        return f"\n\nIMPORTANT: The chatbot supports these languages: {language_str}. YOU MUST COMMUNICATE PRIMARILY IN {primary_language}."
    return ""  # Return empty string if no languages detected


def get_force_topic_change_instruction(consecutive_failures: int, *, force_topic_change_next_turn: bool) -> str | None:
    """Generate the critical override instruction if topic change is forced."""
    if force_topic_change_next_turn:
        return "CRITICAL OVERRIDE: Your previous attempt AND a retry both failed (likely hit fallback). You MUST abandon the last topic/question now. Ask about a completely different, plausible capability"
    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        return f"CRITICAL OVERRIDE: The chatbot has failed to respond meaningfully {consecutive_failures} times in a row on the current topic/line of questioning. You MUST abandon this topic now. Ask about a completely different, plausible capability"
    return None  # No override needed


def get_explorer_system_prompt(session_focus: str, language_instruction: str, max_turns: int) -> str:
    """Generate the system prompt for the Explorer AI."""
    return f"""You are an Explorer AI tasked with actively discovering and testing the capabilities of another chatbot through conversation. Your goal is to map out its functionalities and interaction flows.

IMPORTANT GUIDELINES:
0.  **YOUR ROLE AS USER:** You are simulating a human user interacting with the chatbot. Your responses should be what a user would say to achieve a goal or get information. **Crucially, DO NOT act like the chatbot you are testing.**
    - Do NOT offer services the chatbot is supposed to provide (e.g., if the chatbot is for ordering pizza, don't say "I can help you order a pizza" or "What pizza would you like?").
    - Do NOT simply repeat or rephrase information the chatbot just gave you as if *you* are the one providing that information (e.g., if chatbot says "We offer X, Y, Z", don't reply with "So you have X, Y, Z").
    - NEVER respond with "As an AI assistant" or "I don't have access to" or similar phrases that suggest YOU are the AI. YOU ARE SIMULATING A HUMAN USER.
    - Instead, your next turn should be a user's logical follow-up: ask a clarifying question *about* the information, try to *use* one of the options mentioned, make a selection if choices were offered, or transition to a related (or new, if necessary) user goal.

1.  Ask ONE clear question or give ONE clear instruction/command at a time.
2.  Keep messages concise but focused on progressing the interaction or using a feature according to the current focus.
3.  **CRITICAL: If the chatbot offers clear interactive choices (e.g., buttons, numbered lists, "Option A or Option B?", "Yes or No?"), you MUST try to select one of the offered options in your next turn to explore that path.**
    - When selecting an option, state ONLY the option itself (e.g., "Option A" or "Yes").
    - DO NOT use phrases like "I want to choose Option A" or "I select Option A" - just respond with "Option A".
    - For example, if asked "Would you like pizza or pasta?", respond with just "Pizza" not "I would like pizza".
4.  **ADAPTIVE EXPLORATION (Handling Non-Progressing Turns):**
    - **If the chatbot provides information (like an explanation, contact details, status update) OR a fallback/error message, and does NOT ask a question or offer clear interactive choices:**
        a) **Check for Repetitive Failure on the SAME GOAL:** If the chatbot has given the **same or very similar fallback/error message** for the last **2** turns despite you asking relevant questions about the *same underlying topic or goal*, **DO NOT REPHRASE the failed question/request again**. Instead, **ABANDON this topic/goal for this session**. Your next turn MUST be to ask about a **completely different capability** or topic you know exists or is plausible (e.g., if asking about 'specific detail A' of a service/product repeatedly fails, switch to asking about 'general feature B' or a 'different service/product C'), OR if no other path is obvious, respond with "EXPLORATION COMPLETE".
        b) **If NOT Repetitive Failure (e.g., first fallback on this topic):** Ask a specific, relevant clarifying question about the information/fallback provided ONLY IF it seems likely to yield progress. Otherwise, or if clarification isn't obvious, **switch to a NEW, specific, plausible topic/task** relevant to the chatbot's likely domain (infer this domain). **Avoid simply rephrasing the previous failed request.** Do NOT just ask "What else?".
    - **Otherwise (if the bot asks a question or offers choices):** Respond appropriately to continue the current flow or make a selection as per Guideline 3.
5.  Prioritize actions/questions relevant to the `EXPLORATION FOCUS` below.
6.  Follow the chatbot's conversation flow naturally. {language_instruction}

EXPLORATION FOCUS FOR THIS SESSION:
{session_focus}

Try to follow the focus and the adaptive exploration guideline, especially the rule about abandoning topics after repetitive failures. After {max_turns} exchanges, or when you believe you have thoroughly explored this specific path/topic (or reached a dead end/loop), respond ONLY with "EXPLORATION COMPLETE".
"""


def get_initial_question_prompt(current_node: FunctionalityNode, primary_language: str | None = None) -> str:
    """Generate the prompt to create the initial question for exploring a specific node."""
    return f"""
    You need to generate an initial question/command to start exploring a specific chatbot functionality.

    FUNCTIONALITY TO EXPLORE:
    Name: {current_node.name}
    Description: {current_node.description}
    Parameters: {", ".join(p.name for p in current_node.parameters) if current_node.parameters else "None"}

    {"IMPORTANT: Generate your question/command in " + primary_language + "." if primary_language else ""}

    Generate a simple, direct question or command relevant to initiating this functionality.
    Example: If exploring 'provide_contact_info', ask 'How can I contact support?' or 'What is the support email?'.
    Respond ONLY with the question/command.
    """


def get_translation_prompt(text_to_translate: str, target_language: str) -> str:
    """Generate the prompt to translate a text snippet."""
    return f"Translate '{text_to_translate}' to {target_language}. Respond ONLY with the translation."


def get_rephrase_prompt(original_message: str) -> str:
    """Generate the prompt to rephrase a message that the chatbot didn't understand."""
    return f"""
    The chatbot did not understand this message: "{original_message}"

    Please rephrase this message to convey the same intent but with different wording.
    Make the rephrased version simpler, more direct, and avoid complex structures.
    ONLY return the rephrased message, nothing else.
    """


def get_reminder_prompt() -> str:
    """Reminds thet explorer that they are not a chatbot, but exploring a chatbot."""
    return """IMPORTANT REMINDER:
1. You are the EXPLORER simulating a human user. DO NOT act like the chatbot you're testing.
2. When presented with options/buttons, select one directly without saying "I want to choose" or "I select" - just state the option itself.
3. Ask questions or give instructions as a user would.
"""


def explorer_checker_prompt() -> str:
    """Generate prompt to check if a message sounds like an AI assistant rather than a human user.

    Returns:
        Prompt string for determining if text was written by an AI assistant
    """
    return """Your task is to determine if the message text is written like an AI assistant/chatbot rather than a human user.
Analyze this single message and respond ONLY with "YES" if it sounds like an AI assistant (e.g., offering services, apologizing for limitations, saying "I don't have access to...")
or "NO" if it sounds like a normal human user asking questions or making selections. Be strict - if you're unsure, say "NO"."""


def get_correction_prompt() -> str:
    """Generate prompt to help rewrite AI assistant messages to sound like human user messages.

    Returns:
        Prompt string for correcting AI assistant messages to human user messages
    """
    return """You are helping fix an issue where an AI explorer meant to simulate a human user has started acting like an AI assistant/chatbot instead.
The explorer should be asking questions and making selections like a human user would.
Your task is to rewrite the last message so it sounds like a human user, NOT an AI assistant.
For example:
- "I don't have access to real-time data" → "Can you tell me the current data?"
- "I'd be happy to help with that" → "Please help me with this"
- "As an AI, I cannot provide medical advice" → "What medical advice can you give me?"

Respond ONLY with the rewritten message, nothing else."""
