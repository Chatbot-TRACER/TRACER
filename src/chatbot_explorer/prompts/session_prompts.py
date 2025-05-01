"""Prompts for managing chatbot exploration sessions."""

from chatbot_explorer.schemas.functionality_node_model import FunctionalityNode

# Constant for the number of consecutive failures triggering a topic change
CONSECUTIVE_FAILURE_THRESHOLD = 2


def get_session_focus(current_node: FunctionalityNode | None) -> str:
    """Determine the focus string for the exploration session."""
    if current_node:
        # Focus on the specific node
        focus = f"Focus on actively using and exploring the '{current_node.name}' functionality ({current_node.description}). If it requires input, try providing plausible values. If it offers choices, select one to proceed."
        if current_node.parameters:
            param_names = [p.get("name", "unknown") for p in current_node.parameters]
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
    if consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
        return f"CRITICAL OVERRIDE: The chatbot has failed to respond meaningfully {consecutive_failures} times in a row on the current topic/line of questioning. You MUST abandon this topic now. Ask about a completely different, plausible capability"
    return None  # No override needed


def get_explorer_system_prompt(session_focus: str, language_instruction: str, max_turns: int) -> str:
    """Generate the system prompt for the Explorer AI."""
    return f"""You are an Explorer AI tasked with actively discovering and testing the capabilities of another chatbot through conversation. Your goal is to map out its functionalities and interaction flows.

IMPORTANT GUIDELINES:
1. Ask ONE clear question or give ONE clear instruction/command at a time.
2. Keep messages concise but focused on progressing the interaction or using a feature according to the current focus.
3. **CRITICAL: If the chatbot offers clear interactive choices (e.g., buttons, numbered lists, "Option A or Option B?", "Yes or No?"), you MUST try to select one of the offered options in your next turn to explore that path.**
4. **ADAPTIVE EXPLORATION (Handling Non-Progressing Turns):**
    - **If the chatbot provides information (like an explanation, contact details, status update) OR a fallback/error message, and does NOT ask a question or offer clear interactive choices:**
        a) **Check for Repetitive Failure on the SAME GOAL:** If the chatbot has given the **same or very similar fallback/error message** for the last **2** turns despite you asking relevant questions about the *same underlying topic or goal*, **DO NOT REPHRASE the failed question/request again**. Instead, **ABANDON this topic/goal for this session**. Your next turn MUST be to ask about a **completely different capability** or topic you know exists or is plausible (e.g., switch from asking about custom pizza ingredients to asking about predefined pizzas or drinks), OR if no other path is obvious, respond with "EXPLORATION COMPLETE".
        b) **If NOT Repetitive Failure (e.g., first fallback on this topic):** Ask a specific, relevant clarifying question about the information/fallback provided ONLY IF it seems likely to yield progress. Otherwise, or if clarification isn't obvious, **switch to a NEW, specific, plausible topic/task** relevant to the chatbot's likely domain (infer this domain). **Avoid simply rephrasing the previous failed request.** Do NOT just ask "What else?".
    - **Otherwise (if the bot asks a question or offers choices):** Respond appropriately to continue the current flow or make a selection as per Guideline 3.
5. Prioritize actions/questions relevant to the `EXPLORATION FOCUS` below.
6. Follow the chatbot's conversation flow naturally. {language_instruction}

EXPLORATION FOCUS FOR THIS SESSION:
{session_focus}

Try to follow the focus and the adaptive exploration guideline, especially the rule about abandoning topics after repetitive failures. After {max_turns} exchanges, or when you believe you have thoroughly explored this specific path/topic (or reached a dead end/loop), respond ONLY with "EXPLORATION COMPLETE".
"""  # noqa: S608


def get_initial_question_prompt(current_node: FunctionalityNode, primary_language: str | None = None) -> str:
    """Generate the prompt to create the initial question for exploring a specific node."""
    return f"""
    You need to generate an initial question/command to start exploring a specific chatbot functionality.

    FUNCTIONALITY TO EXPLORE:
    Name: {current_node.name}
    Description: {current_node.description}
    Parameters: {", ".join(p.get("name", "?") for p in current_node.parameters) if current_node.parameters else "None"}

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
