import sys
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
import argparse  # Add this import

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from chatbot_connectors import ChatbotTaskyto


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Chatbot Explorer - Discover functionalities of another chatbot"
    )

    default_sessions = 3
    default_turns = 8
    default_url = "http://localhost:5000"
    default_model = "gpt-4o-mini"
    default_file = "discovered_functionalities.txt"

    parser.add_argument(
        "-s",
        "--sessions",
        type=int,
        default=default_sessions,
        help=f"Number of exploration sessions (default: {default_sessions})",
    )

    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        default=default_turns,
        help=f"Maximum turns per session (default: {default_turns})",
    )

    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=default_url,
        help=f"Chatbot URL to explore (default: {default_url})",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=default_model,
        help=f"OpenAI model to use (default: {default_model})",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=default_file,
        help=f"Output file to save discovered functionalities (default: {default_file})",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Display configuration
    print("=== Chatbot Explorer Configuration ===")
    print(f"Chatbot URL: {args.url}")
    print(f"Exploration sessions: {args.sessions}")
    print(f"Max turns per session: {args.turns}")
    print(f"Using model: {args.model}")
    print(f"Output file: {args.output}")
    print("====================================")

    # Use the parameters from args
    chatbot_url = args.url
    max_sessions = args.sessions
    max_turns = args.turns
    model_name = args.model
    output_file = args.output

    # Track multiple conversation sessions
    conversation_sessions = []

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        conversation_history: list
        discovered_functionalities: list
        discovered_limitations: list
        current_session: int
        # Bool field that determines if the exploration phase is completed
        exploration_finished: bool
        conversation_goals: list

    llm = ChatOpenAI(model=model_name)

    def goal_generator(state: State):
        """Generate conversation goals based on discovered functionalities"""
        if state["exploration_finished"] and state["discovered_functionalities"]:
            print("\n--- Generating conversation goals ---")

            # Generate goals using the existing function
            profiles_with_goals = generate_user_profiles_and_goals(
                state["discovered_functionalities"],
                state["discovered_limitations"],
                llm
            )

            # Return updated state with goals
            return {
                "messages": state["messages"],
                "conversation_goals": profiles_with_goals
            }
        return {"messages": state["messages"]}

    # This node will talk with the other chatbot and figure out its functionalities
    def explorer(state: State):
        # Only process if we're in exploration phase
        if not state["exploration_finished"]:
            return {"messages": [llm.invoke(state["messages"])], "explored": True}
        # If not just return the messages
        return {"messages": state["messages"]}

    # This node will analyze the functionalities with a given conversation
    def analyzer(state: State):
        if state["exploration_finished"]:
            # Create prompt for analyzer
            analyzer_prompt = f"""
            You are a Functionality Analyzer tasked with extracting a comprehensive list of functionalities from conversation histories.

            Below are transcripts from {len(state["conversation_history"])} different conversation sessions with the same chatbot.

            Your task is to:
            1. Extract all distinct functionalities the chatbot appears to have
            2. Provide a clear, structured list with descriptions
            3. Note any limitations or constraints you observed

            CONVERSATION HISTORY:
            {state["conversation_history"]}

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

            analysis_result = llm.invoke(analyzer_prompt)
            analysis_content = analysis_result.content
            functionalities = extract_functionalities(analysis_content)
            limitations = extract_limitations(analysis_content)

            return {
                "messages": state["messages"] + [analysis_result],
                "discovered_functionalities": functionalities,
                "discovered_limitations": limitations,
            }
        return {"messages": state["messages"]}

    def extract_functionalities(analysis_text):
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

    # This function only extracts limitations (keep your existing one)
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

    # Set up the graph
    graph_builder = StateGraph(State)

    graph_builder.add_node("explorer", explorer)
    graph_builder.add_node("analyzer", analyzer)
    graph_builder.add_node("goal_generator", goal_generator)

    graph_builder.set_entry_point("explorer")
    graph_builder.add_edge("explorer", "analyzer")
    graph_builder.add_edge("analyzer", "goal_generator")
    graph_builder.set_finish_point("goal_generator")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}

    the_chatbot = ChatbotTaskyto(chatbot_url)

    # Session management loop
    for session_num in range(max_sessions):
        print(
            f"\n--- Starting Exploration Session {session_num + 1}/{max_sessions} ---"
        )

        # Reset conversation history for this session
        conversation_history = [
            {
                "role": "system",
                "content": f"""You are an Explorer AI tasked with learning about another chatbot you're interacting with.

IMPORTANT GUIDELINES:
1. Ask ONE simple question at a time - the chatbot gets confused by multiple questions
2. Keep your messages short and direct
3. When the chatbot indicates it didn't understand, simplify your language further
4. Follow the chatbot's conversation flow and adapt to its capabilities

EXPLORATION FOCUS FOR SESSION {session_num + 1}:
{
                    "Ask about ordering process and customization options"
                    if session_num == 0
                    else "Focus on information retrieval capabilities like hours, prices, etc."
                    if session_num == 1
                    else "Explore error handling and limitations of the chatbot"
                }

Your goal is to understand the chatbot's capabilities through direct, simple interactions.
After approximately 10 exchanges, or when you feel you've explored this path thoroughly, say "EXPLORATION COMPLETE".
""",
            }
        ]

        print("Starting session")
        is_ok, chatbot_message = the_chatbot.execute_starter_chatbot()
        print(f"\nChatbot: {chatbot_message}")

        # Some ideas on how to start the conversation
        first_questions = [
            "Hello! How can you help me?",
            "What are your your functionalities?",
            "What can you do?",
        ]

        initial_question = (
            first_questions[session_num]
            if session_num < len(first_questions)
            else "Hello! What can you tell me about yourself?"
        )

        conversation_history.append({"role": "user", "content": chatbot_message})
        conversation_history.append({"role": "assistant", "content": initial_question})

        # Conduct the conversation for this session
        is_ok, chatbot_message = the_chatbot.execute_with_input(
            conversation_history[-1]["content"]
        )
        print(f"\nExplorer: {conversation_history[-1]['content']}")
        print(f"\nChatbot: {chatbot_message}")

        # Main conversation loop for this session
        turn_count = 0

        while True:
            turn_count += 1

            # Exit conditions
            if turn_count >= max_turns:
                print(
                    f"\nReached maximum turns ({max_turns}). Ending session {session_num + 1}."
                )
                break

            # Add the chatbot message to the conversation history
            conversation_history.append({"role": "user", "content": chatbot_message})

            # Process through LangGraph with full history context
            explorer_response = None
            for event in graph.stream(
                {
                    "messages": conversation_history,
                    "conversation_history": [],
                    "discovered_functionalities": [],
                    "current_session": session_num,
                    "exploration_finished": False,
                    "conversation_goas": [],
                },
                config=config,
            ):
                for value in event.values():
                    latest_message = value["messages"][-1]
                    explorer_response = latest_message.content

            print(f"\nExplorer: {explorer_response}")

            # Check for session completion
            if "EXPLORATION COMPLETE" in explorer_response.upper():
                print(f"\nExplorer has finished session {session_num + 1} exploration.")
                break

            # Add the explorer response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": explorer_response}
            )

            # Send explorer response back to Chatbot
            is_ok, chatbot_message = the_chatbot.execute_with_input(explorer_response)

            if not is_ok:
                print("\nError communicating with chatbot. Ending session.")
                break

            print(f"\nChatbot: {chatbot_message}")

            # Check for exit conditions from the chatbot
            if chatbot_message.lower() in ["quit", "exit", "q", "bye", "goodbye"]:
                print("Chatbot ended the conversation. Ending session.")
                break

        # After session ends, save the conversation history
        print(f"\nSession {session_num + 1} complete with {turn_count} exchanges")
        conversation_sessions.append(conversation_history)

    # Fix the analysis invocation by adding the config parameter
    print("\n--- All exploration sessions complete. Analyzing results... ---")

    # Create state for analysis
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            }
        ],
        "conversation_history": conversation_sessions,
        "discovered_functionalities": [],
        "discovered_limitations": [],
        "current_session": max_sessions,
        "exploration_finished": True,
        "conversation_goals": [],
    }

    # Execute the analysis
    config = {"configurable": {"thread_id": "analysis_session"}}
    result = graph.invoke(analysis_state, config=config)

    # Display results with error handling for the missing key
    print("\n=== CHATBOT FUNCTIONALITY ANALYSIS ===")
    print("\n## FUNCTIONALITIES")
    for i, func in enumerate(result.get("discovered_functionalities", []), 1):
        print(f"{i}. {func}")

    print("\n## LIMITATIONS")
    if "discovered_limitations" in result:
        for i, limitation in enumerate(result["discovered_limitations"], 1):
            print(f"{i}. {limitation}")
    else:
        print("No limitations discovered.")

    # Save results with error handling
    with open(output_file, "w") as f:
        f.write("## FUNCTIONALITIES\n")
        for func in result.get("discovered_functionalities", []):
            f.write(f"- {func}\n")
        f.write("\n## LIMITATIONS\n")
        if "discovered_limitations" in result:
            for limitation in result["discovered_limitations"]:
                f.write(f"- {limitation}\n")
        else:
            f.write("- No limitations discovered.\n")

    # Generate user profiles and goals
    print("\n--- User profiles and goals from analysis ---")
    if "conversation_goals" in result and result["conversation_goals"]:
        for profile in result["conversation_goals"]:
            print(f"\nProfile: {profile['name']}")
            print(f"Description: {profile['description']}")
            print("\nFunctionalities:")
            for func in profile["functionalities"]:
                print(f"- {func}")
            print("\nGoals:")
            for goal in profile["goals"]:
                print(f"- {goal}")
    else:
        print("No conversation goals were generated.")


def generate_user_profiles_and_goals(
    functionalities, limitations, llm, output_dir="profiles"
):
    """
    Group functionalities into logical user profiles and generate coherent goal sets
    for individual conversations
    """
    # First, create the output directory if it doesn't exist
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ask the LLM to identify distinct conversation scenarios
    grouping_prompt = f"""
    Based on these chatbot functionalities:
    {", ".join(functionalities)}

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

        DO NOT include variable definitions - just use {{varname}} placeholders (important, two curly braces to open and two to close).
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


if __name__ == "__main__":
    main()
