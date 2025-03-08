from typing import Annotated, Dict, List, Any
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State for the LangGraph flow"""

    messages: Annotated[list, add_messages]
    conversation_history: list
    discovered_functionalities: list
    discovered_limitations: list
    current_session: int
    exploration_finished: bool
    conversation_goals: list
    supported_languages: list


class ChatbotExplorer:
    """Manages exploration of target chatbots using LangGraph."""

    def __init__(self, model_name: str):
        """Initialize explorer with the given model name."""
        self.llm = ChatOpenAI(model=model_name)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph for exploration."""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("explorer", self._explorer_node)
        graph_builder.add_node("analyzer", self._analyzer_node)
        graph_builder.add_node("goal_generator", self._goal_generator_node)

        # Add edges
        graph_builder.set_entry_point("explorer")
        graph_builder.add_edge("explorer", "analyzer")
        graph_builder.add_edge("analyzer", "goal_generator")
        graph_builder.set_finish_point("goal_generator")

        return graph_builder.compile(checkpointer=self.memory)

    def _explorer_node(self, state: State):
        """Explorer node for interacting with the target chatbot."""
        if not state["exploration_finished"]:
            return {"messages": [self.llm.invoke(state["messages"])], "explored": True}
        return {"messages": state["messages"]}

    def _analyzer_node(self, state: State):
        """Analyzer node for processing chatbot responses."""
        if state["exploration_finished"]:
            # Add language prompt
            language_instruction = ""
            if state["supported_languages"]:
                primary_language = state["supported_languages"][0]
                language_instruction = f"""
IMPORTANT LANGUAGE INSTRUCTION:
- Write all functionality descriptions and limitations in {primary_language}
- KEEP THE HEADINGS (## IDENTIFIED FUNCTIONALITIES, ## LIMITATIONS) IN ENGLISH
- MAINTAIN THE NUMBERED FORMAT (1., 2., etc.) with colons
- Example: "1. [Functionality name]: [Description in {primary_language}]"
"""
            analyzer_prompt = f"""
            You are a Functionality Analyzer tasked with extracting a comprehensive list of functionalities from conversation histories.

            Below are transcripts from {len(state["conversation_history"])} different conversation sessions with the same chatbot.

            Your task is to:
            1. Extract all distinct functionalities the chatbot appears to have
            2. Provide a clear, structured list with descriptions
            3. Note any limitations or constraints you observed

            CONVERSATION HISTORY:
            {state["conversation_history"]}

            {language_instruction}

            FORMAT YOUR RESPONSE AS:
            ## IDENTIFIED FUNCTIONALITIES
            1. [Functionality Name]: [Description]
            2. [Functionality Name]: [Description]
            ...

            ## LIMITATIONS
            - [Limitation 1]
            - [Limitation 2]
            ...
            """

            analysis_result = self.llm.invoke(analyzer_prompt)
            analysis_content = analysis_result.content
            functionalities = self.extract_functionalities(analysis_content)
            limitations = self.extract_limitations(analysis_content)

            return {
                "messages": state["messages"] + [analysis_result],
                "discovered_functionalities": functionalities,
                "discovered_limitations": limitations,
            }
        return {"messages": state["messages"]}

    def _goal_generator_node(self, state: State):
        """Node for generating conversation goals based on discovered functionalities."""
        from .profiles import generate_user_profiles_and_goals

        if state["exploration_finished"] and state["discovered_functionalities"]:
            print("\n--- Generating conversation goals ---")

            profiles_with_goals = generate_user_profiles_and_goals(
                state["discovered_functionalities"],
                state["discovered_limitations"],
                self.llm,
                conversation_history=state["conversation_history"],
                supported_languages=state["supported_languages"],
            )

            return {
                "messages": state["messages"],
                "conversation_goals": profiles_with_goals,
            }
        return {"messages": state["messages"]}

    def extract_functionalities(self, analysis_text):
        """Extract functionalities from the analysis text."""
        functionalities = []

        if "## IDENTIFIED FUNCTIONALITIES" in analysis_text:
            func_section = analysis_text.split("## IDENTIFIED FUNCTIONALITIES")[1]
            if "##" in func_section:
                func_section = func_section.split("##")[0]

            func_lines = [
                line.strip() for line in func_section.split("\n") if line.strip()
            ]
            for line in func_lines:
                if ":" in line and any(char.isdigit() for char in line[:3]):
                    # Extract functionality from numbered list format
                    func_parts = line.split(":", 1)
                    if len(func_parts) > 1:
                        func_name = func_parts[0].strip().split(".", 1)[-1].strip()
                        func_desc = func_parts[1].strip()
                        functionalities.append(f"{func_name}: {func_desc}")

        return functionalities

    def extract_limitations(self, analysis_text):
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

    def run_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Run graph with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.invoke(state, config=config)

    def stream_exploration(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream graph execution with the given state and config."""
        if config is None:
            config = {"configurable": {"thread_id": "1"}}
        return self.graph.stream(state, config=config)


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
