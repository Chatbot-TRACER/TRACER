import uuid
from typing import Any, Dict, List

from chatbot_explorer.agent import (
    ChatbotExplorationAgent,
)


def run_analysis_pipeline(
    explorer_instance: ChatbotExplorationAgent, exploration_results: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Runs the LangGraph analysis pipeline to infer workflow structure and generate user profiles.

    Args:
        explorer_instance: An instance of ChatbotExplorer
        exploration_results: A dictionary containing results from the exploration phase,
                             including 'conversation_sessions', 'root_nodes_dict',
                             'supported_languages', and 'fallback_message'.

    Returns:
        A dictionary containing the key results needed for reporting:
        - 'discovered_functionalities': The final structured functionalities after analysis.
        - 'built_profiles': The list of generated user profiles.
    """
    print("\n--- Preparing to infer complete workflow structure and generate user profiles ---")

    # Prepare initial state for LangGraph analysis using exploration results
    analysis_state = {
        "messages": [
            {
                "role": "system",
                "content": "Analyze the conversation histories to identify functionalities",
            },
        ],
        "conversation_history": exploration_results["conversation_sessions"],
        "discovered_functionalities": exploration_results["root_nodes_dict"],  # Use results from exploration
        "discovered_limitations": [],  # Limitations are not currently extracted during exploration
        "current_session": len(exploration_results["conversation_sessions"]),  # Use actual number of sessions run
        "exploration_finished": True,
        "conversation_goals": [],  # Not used in this flow currently
        "supported_languages": exploration_results["supported_languages"],
        "fallback_message": exploration_results["fallback_message"],
    }

    # -- 1. Infer the workflow structure using the dedicated graph --
    print("\n--- Running workflow structure inference ---")
    structure_graph = explorer_instance._build_structure_graph()

    structure_thread_id = f"structure_analysis_{uuid.uuid4()}"
    structure_result = structure_graph.invoke(
        analysis_state, config={"configurable": {"thread_id": structure_thread_id}}
    )

    # Update state with the refined structure from the structure graph
    analysis_state["discovered_functionalities"] = structure_result["discovered_functionalities"]

    # Store the workflow structure for usage in profile generation
    workflow_structure = structure_result["discovered_functionalities"]

    # -- 2. Generate user profiles based on the inferred structure --
    print("\n--- Generating user profiles ---")
    profile_graph = explorer_instance._build_profile_generation_graph()

    # Add workflow structure to the state for the profile generation graph
    analysis_state["workflow_structure"] = workflow_structure

    profile_thread_id = f"profile_analysis_{uuid.uuid4()}"
    profile_result = profile_graph.invoke(analysis_state, config={"configurable": {"thread_id": profile_thread_id}})

    # Extract the final generated profiles
    generated_profiles = profile_result.get("built_profiles", [])

    print(f"--- Analysis complete, {len(generated_profiles)} profiles generated ---")

    # Return the required results for main.py
    return {
        "discovered_functionalities": workflow_structure,
        "built_profiles": generated_profiles,
    }
