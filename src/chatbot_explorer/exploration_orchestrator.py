from typing import Any, Dict, List, Optional, Set

from .explorer import ChatbotExplorer
from .functionality_node import FunctionalityNode
from .session import run_exploration_session
from .utils.conversation.fallback_detection import extract_fallback_message
from .utils.conversation.language_detection import extract_supported_languages


def run_exploration_phase(
    explorer_instance: ChatbotExplorer,
    chatbot_connector,
    max_sessions: int,
    max_turns: int,
) -> Dict[str, Any]:
    """Runs the initial probing and the main exploration loop.

    Args:
        explorer_instance: The ChatbotExplorer instance (provides LLM).
        chatbot_connector: An instance of a chatbot connector class.
        max_sessions (int): Maximum number of exploration sessions to run.
        max_turns (int): Maximum turns per exploration session.

    Returns:
        Dict[str, Any]: A dictionary containing exploration results:
                        - conversation_sessions: List of conversation histories.
                        - root_nodes_dict: List of root FunctionalityNode objects as dicts.
                        - supported_languages: List of detected languages.
                        - fallback_message: Detected fallback message string.
    """
    # Initialize results and state tracking
    conversation_sessions: List[List[Dict[str, str]]] = []
    supported_languages: List[str] = []
    fallback_message: Optional[str] = None
    root_nodes: List[FunctionalityNode] = []
    pending_nodes: List[FunctionalityNode] = []
    explored_nodes: Set[str] = set()

    # --- Initial Language Detection ---
    print("\n--- Probing Chatbot Language ---")
    initial_probe_query = "Hello"  # Simple initial query
    is_ok, probe_response = chatbot_connector.execute_with_input(initial_probe_query)
    if is_ok and probe_response:
        print(f"   Initial response received: '{probe_response[:60]}...'")
        try:
            # Attempt language detection via LLM (using explorer's LLM)
            detected_langs = extract_supported_languages(probe_response, explorer_instance.llm)
            if detected_langs:
                supported_languages = detected_langs
                print(f"   Detected initial language(s): {supported_languages}")
            else:
                print("   Could not detect language from initial probe, defaulting to English.")
        except Exception as lang_e:
            print(f"   Error during initial language detection: {lang_e}. Defaulting to English.")
    else:
        print("   Could not get initial response from chatbot for language probe. Defaulting to English.")
    # --- End Initial Language Detection ---

    # --- Initial Fallback Message Detection ---
    print("\n--- Attempting to detect chatbot fallback message ---")
    try:
        # Call the function using explorer's LLM and the connector
        fallback_message = extract_fallback_message(chatbot_connector, explorer_instance.llm)
        if fallback_message:
            print(f'   Detected fallback message: "{fallback_message[:50]}..."')
        else:
            print("   Could not detect a fallback message.")
    except Exception as fb_e:
        print(f"   Error during fallback detection: {fb_e}. Proceeding without fallback.")
        fallback_message = None
    # --- End Fallback Message Detection ---

    # --- Exploration Loop ---
    session_num = 0
    while session_num < max_sessions:
        current_session_index = session_num
        print(f"\n=== Starting Session {current_session_index + 1}/{max_sessions} ===")

        explore_node = None  # Node to focus on this session
        session_type_log = "General Exploration"

        if pending_nodes:
            # Prioritize exploring specific nodes from the queue
            explore_node = pending_nodes.pop(0)
            # Double-check if node was already explored
            if explore_node.name in explored_nodes:
                print(f"--- Skipping already explored node: '{explore_node.name}' ---")
                session_num += 1  # Consume a session slot
                continue
            session_type_log = f"Exploring functionality '{explore_node.name}'"
        elif session_num > 0:  # If queue is empty after session 0, perform general exploration
            print("   Pending nodes queue is empty. Performing general exploration.")
        # Else: Session 0 and queue is empty is the initial state.

        print(f"   Session Type: {session_type_log}")

        # Execute one exploration session using the imported function
        (
            conversation_history,
            updated_roots,
            updated_pending,
            updated_explored,
        ) = run_exploration_session(
            current_session_index,
            max_sessions,
            max_turns,
            explorer_instance,  # Pass explorer instance (contains self.llm)
            chatbot_connector,
            fallback_message=fallback_message,
            current_node=explore_node,  # None for general exploration
            root_nodes=root_nodes,
            pending_nodes=pending_nodes,
            explored_nodes=explored_nodes,
            supported_languages=supported_languages,
        )

        # Aggregate results
        conversation_sessions.append(conversation_history)
        root_nodes = updated_roots
        pending_nodes = updated_pending  # Includes newly discovered nodes
        explored_nodes = updated_explored

        # Move to the next session
        session_num += 1
    # --- End Exploration Loop ---

    # --- Post-Exploration Summary ---
    if session_num == max_sessions:
        print(f"\n=== Completed {max_sessions} exploration sessions ===")
        if pending_nodes:
            print(f"   NOTE: {len(pending_nodes)} nodes still remain in the pending queue.")
        else:
            print("   All discovered nodes were explored.")
    else:
        print(f"\n--- WARNING: Exploration stopped unexpectedly after {session_num} sessions. ---")

    print(f"Discovered {len(root_nodes)} root functionalities after exploration.")

    # Convert FunctionalityNode objects to dictionaries for the return value
    functionality_dicts = [node.to_dict() for node in root_nodes]

    # Return the collected results
    return {
        "conversation_sessions": conversation_sessions,
        "root_nodes_dict": functionality_dicts,  # Renamed for clarity
        "supported_languages": supported_languages,
        "fallback_message": fallback_message,
    }
