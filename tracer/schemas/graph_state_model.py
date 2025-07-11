"""Defines the models representing the LangGraph graph state."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """Holds the state for the graph.

    This class holds all the necessary information about the current state of the
    chatbot exploration process, including chat messages, conversation history,
    discovered functionalities and limitations, session information, exploration status,
    conversation goals, supported languages, generated profiles, fallback message,
    chatbot type, and workflow structure.

    Attributes:
        messages (Annotated[list, add_messages]): A list of chat messages.
        conversation_history (list): A list containing the history of all conversation sessions.
        discovered_functionalities (List[Dict[str, Any]]): A list of dictionaries, each representing a discovered functionality.
        discovered_limitations (list): A list of discovered limitations of the chatbot.
        current_session (int): The current session number.
        exploration_finished (bool): A flag indicating whether the exploration is finished.
        conversation_goals (list): A list of goals for generating profiles.
        supported_languages (list): A list of languages supported by the chatbot.
        built_profiles (list): A list of generated YAML profiles (as dictionaries or strings).
        fallback_message (str): The chatbot's fallback message.
        chatbot_type (str): The type of chatbot, which can be "transactional", "informational", or "unknown".
        workflow_structure (List[Dict[str, Any]]): A list of dictionaries representing the workflow structure for profile generation.
        nested_forward (bool): Whether to use nested forward() chaining in variable definitions.
        model (str): The LLM model name to use for profile generation (e.g., "gpt-4o-mini", "gemini-2.0-flash").
    """

    messages: Annotated[list, add_messages]
    conversation_history: list
    discovered_functionalities: list[dict[str, Any]]
    discovered_limitations: list
    current_session: int
    exploration_finished: bool
    conversation_goals: list
    supported_languages: list
    built_profiles: list
    fallback_message: str
    chatbot_type: str
    workflow_structure: list[dict[str, Any]]
    nested_forward: bool
    model: str
