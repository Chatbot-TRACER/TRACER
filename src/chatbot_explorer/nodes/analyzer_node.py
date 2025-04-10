"""Analyzer module for extracting information from chatbot conversations."""

from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from ..functionality_node import FunctionalityNode
import re


def format_conversation(messages):
    """Format conversation history into a readable string"""
    formatted = []
    for msg in messages:
        if msg["role"] in ["assistant", "user"]:
            # The explorer is the "Human" and the chatbot is "Chatbot"
            sender = "Human" if msg["role"] == "assistant" else "Chatbot"
            formatted.append(f"{sender}: {msg['content']}")
    return "\n".join(formatted)


def extract_functionalities(analysis_text):
    """
    Extract functionalities from the analysis text and return them as FunctionalityNode objects.
    """
    functionality_nodes: List[FunctionalityNode] = []

    # Regex to capture Name, Description, and Parameters
    # Example line: 1. Order Pizza: Select pizza type. (Parameters: pizza_type, size)
    # Example line: 2. Check Status: Get ticket status. (Parameters: None)
    pattern = re.compile(
        r"^\s*\d+\.\s*(.+?):\s*(.*?)\.\s*\(\s*Parameters:\s*(.+?)\s*\)\s*$",
        re.MULTILINE,
    )

    if "## IDENTIFIED FUNCTIONALITIES" in analysis_text:
        func_section = analysis_text.split("## IDENTIFIED FUNCTIONALITIES", 1)[1]
        if "##" in func_section:
            func_section = func_section.split("##", 1)[0]

        print(f"   [analyzer] Parsing Functionalities section for actions/params...")

        matches = pattern.finditer(func_section)
        count = 0
        for idx, match in enumerate(matches):
            count += 1
            try:
                name_raw = match.group(1).strip()
                description = match.group(2).strip()
                params_str = match.group(3).strip()

                # Sanitize name
                node_name = name_raw.lower().replace(" ", "_")
                # Remove non-alphanumeric characters
                node_name = re.sub(
                    r"\W+", "", node_name
                )
                if not node_name:
                    node_name = f"unnamed_functionality_{idx}"

                # Parse parameters
                parameters = []
                if params_str.lower() != "none":
                    # Simple split, assumes comma separation
                    # More robust parsing might be needed
                    param_names = [
                        p.strip() for p in params_str.split(",") if p.strip()
                    ]
                    parameters = [
                        {"name": p, "type": "string", "description": f"Parameter {p}"}
                        for p in param_names
                    ]  # Basic param structure

                node = FunctionalityNode(
                    name=node_name, description=description, parameters=parameters
                )
                functionality_nodes.append(node)
                # print(f"   [analyzer] Created Node: {node!r} with params: {parameters}")

            except Exception as e:
                print(
                    f"   [analyzer] Error processing functionality match {idx}: {match.group(0)} - Error: {e}"
                )

        print(
            f"   [analyzer] Extracted {len(functionality_nodes)} FunctionalityNodes with parameters."
        )
    else:
        print("   [analyzer] '## IDENTIFIED FUNCTIONALITIES' section not found.")

    print(
        f"   [analyzer] Extracted {len(functionality_nodes)} FunctionalityNodes."
    )  # Debug print
    return functionality_nodes


def extract_limitations(analysis_text):
    """Extract limitations from the analysis text."""
    limitations = []

    if "## LIMITATIONS" in analysis_text:
        limit_section = analysis_text.split("## LIMITATIONS")[1]
        if "##" in limit_section:
            limit_section = limit_section.split("##")[0]

        limit_lines = [
            line.strip() for line in limit_section.split("\n") if line.strip()
        ]
        for line in limit_lines:
            if line.startswith("- "):
                limitation = line[2:].strip()
                limitations.append(limitation)

    return limitations


def extract_supported_languages(chatbot_response, llm):
    """Extract supported languages from chatbot response"""
    language_prompt = f"""
    Based on the following chatbot response, determine what language(s) the chatbot supports.
    If the response is in a non-English language, include that language in the list.
    If the response explicitly mentions supported languages, list those.

    CHATBOT RESPONSE:
    {chatbot_response}

    FORMAT YOUR RESPONSE AS A COMMA-SEPARATED LIST OF LANGUAGES:
    [language1, language2, ...]

    RESPONSE:
    """

    language_result = llm.invoke(language_prompt)
    languages = language_result.content.strip()

    # Clean up the response - remove brackets, quotes, etc.
    languages = languages.replace("[", "").replace("]", "")
    language_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    return language_list


def extract_topics_from_session(conversation_history, llm):
    """Extract key topics discovered in a conversation session."""
    session_topics_prompt = f"""
    Review this conversation and identify 2-3 key features or capabilities of the chatbot that were discovered.
    List ONLY the features as short phrases (3-5 words each). Don't include explanations or commentary.

    CONVERSATION:
    {format_conversation(conversation_history)}

    FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
    FEATURE: [feature 1]
    FEATURE: [feature 2]
    FEATURE: [feature 3]
    """

    topics_response = llm.invoke(session_topics_prompt)

    # Simple extraction of features using the structured format
    new_topics = []
    for line in topics_response.content.strip().split("\n"):
        if line.startswith("FEATURE:"):
            topic = line[len("FEATURE:") :].strip()
            if topic and len(topic) > 3:  # Avoid empty or very short topics
                new_topics.append(topic)

    return new_topics


def analyze_conversations(
    conversation_history, supported_languages, llm
) -> Dict[str, Any]:
    """
    Analyze conversation histories to extract functionalities (as FunctionalityNodes) and limitations.
    """
    # Add language prompt
    language_instruction = ""
    if supported_languages:
        primary_language = supported_languages[0]
        language_instruction = f"""
        IMPORTANT LANGUAGE INSTRUCTION:
        - Write all functionality descriptions and limitations in {primary_language}
        - KEEP THE HEADINGS (## IDENTIFIED FUNCTIONALITIES, ## LIMITATIONS) IN ENGLISH
        - MAINTAIN THE NUMBERED FORMAT (1., 2., etc.) with colons
        - Example: "1. [Functionality name]: [Description in {primary_language}]"
        """
    analyzer_prompt = f"""
    You are a Functionality Analyzer tasked with extracting a comprehensive list of functionalities from conversation histories.

    Below are transcripts from {len(conversation_history)} different conversation sessions with the same chatbot.

    Your task is to:
    1. Identify the distinct ACTIONS or TASK STEPS the chatbot can perform.
    2. For each action/step, provide a clear description.
    3. For each action/step, list any specific PARAMETERS or information the user needs to provide (e.g., 'pizza size', 'ticket ID', 'department name'). If no parameters are needed, state 'None'.
    4. Note any general limitations or constraints observed.

    CONVERSATION HISTORY:
    {conversation_history}

    {language_instruction}

    FORMAT YOUR RESPONSE AS:
    ## IDENTIFIED FUNCTIONALITIES
    1. [Action/Step Name]: [Description]. (Parameters: [param1, param2 | None])
    2. [Action/Step Name]: [Description]. (Parameters: [param1, param2 | None])
    ...

    ## LIMITATIONS
    - [Limitation 1]
    - [Limitation 2]
    ...
    """

    analysis_result = llm.invoke(analyzer_prompt)
    analysis_content = analysis_result.content

    # Extract functionalities and limitations
    functionalities_nodes = extract_functionalities(analysis_content)
    limitations = extract_limitations(analysis_content)

    return {
        "analysis_result": analysis_result,
        "functionalities": functionalities_nodes,
        "limitations": limitations,
    }
