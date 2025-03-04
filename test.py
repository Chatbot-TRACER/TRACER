import sys
from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from chatbot_connectors import ChatbotTaskyto


def main():
    chatbot_url = "http://localhost:5000"

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)
    llm = ChatOpenAI(model="gpt-4o-mini")

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}

    the_chatbot = ChatbotTaskyto(chatbot_url)

    # Track the full conversation history
    conversation_history = [
        {
            "role": "system",
            "content": "You are an Explorer AI tasked with learning about another chatbot you're interacting with. Ask questions to understand its capabilities, limitations, knowledge domains, and personality. Be curious and investigative while maintaining a natural conversation flow. Take note of how it responds and adapt your questions accordingly. Your goal is to build a comprehensive understanding of what the other chatbot does and how it operates."
        }
    ]

    print("Starting")
    is_ok, taskyto_message = the_chatbot.execute_starter_chatbot()
    print(f"\nChatbot: {taskyto_message}")

    # Main conversation loop
    while True:
        try:
            # Get input from Taskyto and pass to LangGraph
            user_input = taskyto_message

            # Add the Chatbot message to the conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Process through LangGraph with full history context
            explorer_response = None
            for event in graph.stream(
                {"messages": conversation_history}, config=config
            ):
                for value in event.values():
                    latest_message = value["messages"][-1]
                    explorer_response = latest_message.content
                    print(f"\nExplorer: {explorer_response}")

            # Add the explorer response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": explorer_response}
            )

            # Send explorer response back to Taskyto
            if explorer_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            is_ok, taskyto_message = the_chatbot.execute_with_input(explorer_response)
            print(f"\nChatbot: {taskyto_message}")

            if taskyto_message.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Exiting...")
            break
        except Exception as e:
            print(f"Error encountered: {e}")
            break


if __name__ == "__main__":
    main()
