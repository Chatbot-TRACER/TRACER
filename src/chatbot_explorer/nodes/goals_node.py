"""Module to generate the goals and their variables"""

import re
from typing import Dict, Any, List, Optional, Set

VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def ensure_double_curly(text):
    # This pattern finds any {something} that is not already wrapped in double braces.
    pattern = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")
    return pattern.sub(r"{{\1}}", text)


def _build_single_variable_prompt(
    profile_name: str,
    role: str,
    goals_text: str,
    variable_name: str,
    all_other_variables: List[str],
    language_instruction: str,
) -> str:
    """
    Creates the specific text prompt to ask the LLM about one variable.

    Args:
        profile_name: The name of the user profile we're working on.
        role: The user's role description
        goals_text: The list of goals for this profile, formatted as a string.
        variable_name: The exact name of the variable we want defined
        all_other_variables: A list of other variable names found in the same profile.
                             Helps the LLM decide if 'forward(other_var)' makes sense.
        language_instruction: A sentence telling the LLM what language to use for
                              example data values (e.g., "Generate examples in Spanish").

    Returns:
        A big string containing the full prompt to send to the LLM.
    """

    other_vars_text = (
        f"OTHER VARIABLES IN THIS PROFILE: {', '.join(sorted(all_other_variables))}"
        if all_other_variables
        else ""
    )

    prompt = f"""
    Define parameters for the variable '{variable_name}' used in a user simulator profile.

    USER PROFILE CONTEXT:
    Name: {profile_name}
    Role: {role}
    Goals (where the variable might appear):
    {goals_text}
    {other_vars_text} # Provides context for potential forward(other_var) usage.

    {language_instruction}

    Define ONLY the parameters for the variable '{variable_name}' following these guidelines:

    1. Choose ONE appropriate FUNCTION from:
        - default(): assigns ALL data in the list to the variable
        - random(): picks ONE random sample from the list
        - random(X): picks X random samples where X is LESS THAN the total number of items
        - random(rand): picks a random number of random samples
        - another(): picks different samples without repetition each time
        - another(X): picks X different samples without repetition
        - forward(): iterates through each sample one by one
        - forward(other_var): iterates and nests with other_var

    IMPORTANT FUNCTION RESTRICTIONS:
    - DO NOT nest functions (e.g., random(another(3)) is INVALID)
    - DO NOT use random(X) where X equals the total number of items (use default() instead)
    - INT() is NOT a function but a TYPE
    - Use forward() for systematic iteration through all values in order
    - Use random() for picking just one value each time (but there could be repetition)
    - Use another() when you want different values on subsequent uses

    2. Choose the most appropriate TYPE from:
        - string: for text values
        - int: for whole numbers
        - float: for decimal numbers

    3. IMPORTANT: Provide DATA in the correct format:
        - For string variables: use a list of realistic string values, the length of the list should be at least 3 items and at most 10 items, use the amount of data that makes sense for the variable
        - For int variables: ALWAYS use the min/max/step format like this:
            min: 1
            max: 10
            step: 1
        - For float variables: ALWAYS use the min/max/step format OR linspace like this:
            min: 1.0
            max: 5.0
            step: 0.5
            OR
            min: 1.0
            max: 5.0
            linspace: 5

    FORMAT YOUR RESPONSE FOR STRING VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: string
        DATA:
        - value1
        - value2
        - value3

        FORMAT YOUR RESPONSE FOR NUMERIC VARIABLES AS:
        VARIABLE: variable_name
        FUNCTION: function_name()
        TYPE: int/float
        DATA:
        min: 1
        max: 10
        step: 1

    Respond ONLY with the FUNCTION, TYPE, and DATA for the variable '{variable_name}'.
    """
    return prompt


def _parse_single_variable_definition(
    response_content: str, expected_type: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Takes the LLM's text answer for one variable and tries to turn it into a Python dictionary.

    Args:
        response_content: The raw text string that the LLM sent back.
        expected_type: (Optional) If we think we know the type (like 'string'),
                       we can pass it. It's mostly just for printing a warning
                       if the LLM gives something different. Defaults to None.

    Returns:
        A dictionary with 'function', 'type', and 'data' keys if parsing worked
        (e.g., {'function': 'random()', 'type': 'string', 'data': ['A', 'B']}).
        Returns None if the LLM's response was messed up or missing required parts.
    """
    definition = {}
    data_lines = []
    in_data_section = False
    parsed_type = None

    lines = response_content.strip().split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if line.startswith("FUNCTION:"):
            definition["function"] = line[len("FUNCTION:") :].strip()
            in_data_section = False
        elif line.startswith("TYPE:"):
            parsed_type = line[len("TYPE:") :].strip()
            definition["type"] = parsed_type
            in_data_section = False
            # If expected_type is provided, check for mismatch early
            if expected_type and parsed_type != expected_type:
                print(
                    f"Warning: LLM returned type '{parsed_type}' but expected '{expected_type}'. Will attempt to parse as '{parsed_type}'."
                )
        elif line.startswith("DATA:"):
            in_data_section = True
            # Determine if data will be list (string) or dict (numeric)
            # Peek at the next line if available to guess format
            next_line_index = i + 1
            if next_line_index < len(lines):
                next_line = lines[next_line_index].strip()
                if ":" in next_line and not next_line.startswith("-"):
                    # Numeric (i.e. min: 1.0 max: ...)
                    definition["data"] = {}
                else:
                    # String (i.e. - value1)
                    definition["data"] = []
            else:
                # Default to list if unsure
                definition["data"] = []
        elif in_data_section:
            data_lines.append(line)

    if (
        not definition.get("function")
        or not definition.get("type")
        or "data" not in definition
    ):
        print(
            f"Warning: Failed to parse essential fields (FUNCTION, TYPE, DATA) from LLM response:\n{response_content}"
        )
        return None  # Essential fields missing

    # --- Post-process DATA based on TYPE ---
    data_type = definition.get("type")
    raw_data = definition["data"]  # This is either [] or {} initially

    if data_type == "string":
        # Expecting a list
        if not isinstance(raw_data, list):
            raw_data = []  # Ensure it's a list
        for item_line in data_lines:
            if item_line.startswith("- "):
                value = item_line[2:].strip().strip("'\"")
                raw_data.append(value)
        if not raw_data:
            print(
                f"Warning: String variable data is empty. LLM response:\n{response_content}"
            )
            return None  # String needs data
        definition["data"] = raw_data

    elif data_type in ["int", "float"]:
        # Expecting a dict
        if not isinstance(raw_data, dict):
            raw_data = {}  # Ensure it's a dict
        for item_line in data_lines:
            if ":" in item_line:
                key, value_str = item_line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()
                try:
                    value = int(value_str) if data_type == "int" else float(value_str)
                    raw_data[key] = value
                except ValueError:
                    print(
                        f"Warning: Could not parse numeric value for key '{key}': '{value_str}'. Skipping."
                    )
        # Validate numeric data structure
        if not (
            "min" in raw_data
            and "max" in raw_data
            and ("step" in raw_data or "linspace" in raw_data)
        ):
            print(
                f"Warning: Numeric variable data missing min/max/step or min/max/linspace. LLM response:\n{response_content}"
            )
            # Allow partial definition for potential fixing later? Or return None? Let's return None for now.
            return None
        definition["data"] = raw_data
    else:
        print(
            f"Warning: Unknown variable type '{data_type}'. Cannot parse data correctly."
        )
        return None  # Unknown type

    # Basic validation
    if (
        not definition.get("function")
        or not definition.get("type")
        or not definition.get("data")
    ):
        print(
            f"Warning: Post-parsing validation failed. Missing fields or empty data. Parsed: {definition}"
        )
        return None

    return definition


def generate_variable_definitions(
    profiles, llm, supported_languages=None, max_retries=3
):
    """
    Goes through user profiles, finds variables like {{this}} in their goals,
    and asks the LLM to define them (type, function, data).

    Args:
        profiles: A list where each item is a dictionary representing a user profile.
                  Needs to have a 'goals' key containing a list of strings.
        llm: The language model object (like ChatOpenAI) used to get definitions.
        supported_languages: (Optional) A list of languages the chatbot knows.
                             The first one is used for examples. Defaults to None.
        max_retries: (Optional) How many times to try asking the LLM again if
                     parsing its response fails for a variable. Defaults to 3.

    Returns:
        The same list of profiles that was passed in, but now the 'goals' list
        inside each profile dictionary also contains the variable definition
        dictionaries (e.g., {'variable_name': {'function': ..., 'type': ..., 'data': ...}}).
    """
    primary_language = ""
    language_instruction = ""
    if supported_languages and len(supported_languages) > 0:
        primary_language = supported_languages[0]
        language_instruction = (
            f"Generate examples/values in {primary_language} where appropriate."
        )

    # Process each profile
    for profile in profiles:
        # Extract all unique variables from the goals
        all_variables: Set[str] = set()
        goals_list = profile.get("goals", [])
        if not isinstance(goals_list, list):  # Ensure goals is a list
            print(
                f"Warning: Profile '{profile.get('name', 'Unnamed')}' has invalid 'goals' format. Skipping variable definition."
            )
            continue

        for goal in goals_list:
            if isinstance(goal, str):
                variables_in_goal = VARIABLE_PATTERN.findall(goal)
                all_variables.update(variables_in_goal)
            # Ignore non-string goals (like existing variable definitions if run multiple times)

        if not all_variables:
            print(
                f"Info: No variables found in goals for profile '{profile.get('name', 'Unnamed')}'."
            )
            continue

        print(
            f"\n--- Defining variables for profile: {profile.get('name', 'Unnamed')} ---"
        )
        print(f"   Found variables: {', '.join(sorted(all_variables))}")

        goals_text = ""
        for goal in goals_list:
            if isinstance(goal, str):
                goals_text += f"- {goal}\n"  # Only include string goals in the context

        # Define each variable individually
        for variable_name in sorted(all_variables):  # Sort for consistent order
            print(f"   Defining variable: '{variable_name}'...")
            other_variables = list(all_variables - {variable_name})
            parsed_def = None

            for attempt in range(max_retries):
                prompt = _build_single_variable_prompt(
                    profile.get("name", "Unnamed Profile"),
                    profile.get("role", "Unknown Role"),
                    goals_text,
                    variable_name,
                    other_variables,
                    language_instruction,
                )
                try:
                    response = llm.invoke(prompt)
                    response_content = response.content

                    parsed_def = _parse_single_variable_definition(response_content)

                    if parsed_def:
                        print(
                            f"      Successfully parsed definition for '{variable_name}'."
                        )
                        break  # Success
                    else:
                        print(
                            f"      Warning: Failed to parse definition for '{variable_name}' on attempt {attempt + 1}. Retrying..."
                        )

                except Exception as e:
                    print(
                        f"      Error during LLM call or parsing for '{variable_name}' on attempt {attempt + 1}: {e}. Retrying..."
                    )
                    parsed_def = None  # Ensure reset on error

            # Store the result (or handle failure)
            if parsed_def:
                # Add the definition as a dictionary under the variable name key
                # Check if a definition already exists (e.g., from a previous run or manual entry)
                existing_def_index = -1
                for i, goal_item in enumerate(goals_list):
                    if isinstance(goal_item, dict) and variable_name in goal_item:
                        existing_def_index = i
                        break

                if existing_def_index != -1:
                    print(f"      Updating existing definition for '{variable_name}'.")
                    goals_list[existing_def_index] = {variable_name: parsed_def}
                else:
                    print(f"      Adding new definition for '{variable_name}'.")
                    goals_list.append(
                        {variable_name: parsed_def}
                    )  # Append new definition dict

            else:
                print(
                    f"      ERROR: Failed to generate valid definition for variable '{variable_name}' after {max_retries} attempts."
                )

        # Update the profile's goals list with the added/updated definitions
        profile["goals"] = goals_list

    return profiles


def generate_context(
    profile_name: str,
    role: str,
    sanitized_goals: List[str],
    path_nodes: List[Dict[str, Any]],
    llm,
    language_instruction: str,
) -> List[str]:
    """
    Generate additional context points for a user simulator based on a specific profile/path.

    Args:
        profile_name: The name of the profile being generated.
        role: The role assigned to the user for this profile.
        sanitized_goals: List of goal strings with variables replaced by placeholders.
        path_nodes: The list of node dictionaries representing the specific path.
        llm: The language model instance.
        language_instruction: Instruction for the LLM regarding language.

    Returns:
        A list of generated context strings. Returns an empty list on error.
    """
    print(f"   Generating context for profile: '{profile_name}'...")

    # Format node info for the prompt
    nodes_info_str = "\n".join(
        [
            f"- {node.get('name', 'N/A')}: {node.get('description', 'N/A')[:100]}..."
            for node in path_nodes
        ]
    )
    if not nodes_info_str:
        nodes_info_str = "N/A"  # Empty path case

    # Build the LLM prompt with all our info
    context_prompt = f"""
    Create 2-3 SHORT context points for a user simulator interacting with a chatbot.
    Each point should be a separate piece of background information or context that helps the simulator act according to its role and goals for this specific scenario.

    CONVERSATION SCENARIO: {profile_name}
    USER ROLE: {role}
    USER GOALS (generalized):
    {", ".join(sanitized_goals)}

    RELEVANT FUNCTIONALITY/TOPIC(S) IN THIS SCENARIO:
    {nodes_info_str}

    {language_instruction}

    GUIDELINES:
    1. Write 2-3 SEPARATE short context points, each 1-2 sentences only.
    2. Each point should focus on ONE aspect (background info, knowledge, motivation) relevant to the role, goals, and functionalities listed above.
    3. NEVER include variables like {{date}} or {{amount}} - use specific examples instead if needed for realism.
    4. Keep each point brief and focused.
    5. Make each point distinctly different from the others.

    Examples of GOOD context points:
    - "You tried calling the office yesterday but no one answered."
    - "You need to finish this task before your meeting at 3pm."
    - "Your colleague mentioned this service was very reliable."

    FORMAT YOUR RESPONSE WITH ONE CONTEXT POINT PER LINE:
    - First context point
    - Second context point
    - Third context point (optional)

    IMPORTANT: Each line should start with "- " and be a standalone piece of context. Respond ONLY with the context points.
    """

    context_entries = []
    try:
        # Get response from LLM
        context_response = llm.invoke(context_prompt)
        context_content = context_response.content.strip()
        print(f"      LLM response for context:\n{context_content[:200]}...")

        # Extract context points from response
        for line in context_content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                entry = line[2:].strip().strip("\"'")  # Remove the dash and quotes
                if entry:
                    context_entries.append(entry)

        # Show warning if no context was found
        if not context_entries:
            print(f"   Warning: No context points parsed for profile '{profile_name}'.")

    except Exception as e:
        # Error handling
        print(f"   Error generating context for profile '{profile_name}': {e}")
        return []

    print(f"      Generated {len(context_entries)} context points.")
    return context_entries


def generate_outputs(
    profile_name: str,
    role: str,
    sanitized_goals: List[str],
    path_nodes: List[Dict[str, Any]],
    llm,
    language_instruction: str,
) -> List[Dict[str, Any]]:
    """
    Generate output fields to extract from chatbot responses for a specific profile/path.

    Args:
        profile_name: The name of the profile being generated.
        role: The role assigned to the user for this profile.
        sanitized_goals: List of goal strings with variables replaced by placeholders.
        path_nodes: The list of node dictionaries representing the specific path.
        llm: The language model instance.
        language_instruction: Instruction for the LLM regarding language.

    Returns:
        A list of generated output dictionaries (e.g., [{'output_name': {'type': 'str', 'description': ...}}]).
        Returns an empty list on error.
    """
    print(f"   Generating outputs for profile: '{profile_name}'...")

    # Format node info for prompt
    nodes_info_str = "\n".join(
        [
            f"- {node.get('name', 'N/A')}: {node.get('description', 'N/A')[:100]}..."
            for node in path_nodes
        ]
    )
    if not nodes_info_str:
        nodes_info_str = "N/A"  # Handle empty path

    # Create the prompt for LLM with detailed instructions
    outputs_prompt = f"""
    Identify 2-4 key pieces of INFORMATION THE CHATBOT WILL PROVIDE that should be extracted from its responses for this specific scenario.
    These outputs help the tester validate data from the conversation once it is finished.

    CONVERSATION SCENARIO: {profile_name}
    USER ROLE: {role}
    USER GOALS (generalized):
    {", ".join(sanitized_goals)}

    RELEVANT FUNCTIONALITY/TOPIC(S) IN THIS SCENARIO:
    {nodes_info_str}

    {language_instruction}

    PURPOSE OF OUTPUTS:
    The tester needs to extract specific data points from the chatbot's responses to validate
    the consistency and performance of the chatbot *within this scenario*. These outputs represent information
    that should appear in the chatbot's messages related to the user's goals and the involved functionalities.

    EXTREMELY IMPORTANT:
    - Outputs must be information THE CHATBOT GIVES TO THE USER, not what the user inputs.
    - Focus on information that validates whether the chatbot is working correctly for this scenario.
    - Choose data points that would appear in the chatbot's responses during this interaction flow.

    Examples of GOOD outputs:
    - total_cost: The final price quoted by the chatbot for a service
    - appointment_time: When the chatbot says an appointment is available
    - confirmation_number: A reference number provided by the chatbot
    - service_hours: Operating hours mentioned by the chatbot

    Examples of BAD outputs (these are user inputs, not chatbot outputs):
    - selected_toppings: This is what the user tells the chatbot, not what we extract
    - requested_appointment: This is the user's input, not chatbot's output
    - user_query: This is what the user asks, not what we extract

    For each output:
    1. Give it a SHORT, descriptive name (use underscores instead of spaces).
    2. Assign an appropriate type from the following options:
       - str: For text data (names, descriptions, IDs)
       - int: For whole numbers
       - float: For decimal numbers
       - date: For calendar dates
       - time: For time values
    3. Write a brief description of what this output represents and how to find it in the chatbot's responses.

    THE OUTPUT STRUCTURE MUST FOLLOW THIS EXACT FORMAT:
    OUTPUT: output_name_1
    TYPE: output_type_1
    DESCRIPTION: brief description of what to extract from chatbot responses

    OUTPUT: output_name_2
    TYPE: output_type_2
    DESCRIPTION: brief description of what to extract from chatbot responses

    PROVIDE 2-4 OUTPUTS, EACH FOLLOWING THIS EXACT FORMAT. Respond ONLY with the outputs.
    """

    outputs_list = []
    try:
        # Get LLM response
        outputs_response = llm.invoke(outputs_prompt)
        response_content = outputs_response.content.strip()
        print(f"      LLM response for outputs:\n{response_content[:200]}...")

        # Need to parse sections for each output
        current_output = None
        current_data = {}

        # Go through response line by line to extract outputs
        for line in response_content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Start of a new output definition
            if line.startswith("OUTPUT:"):
                # Save previous output if we have one
                if current_output and current_data:
                    if "type" in current_data and "description" in current_data:
                        outputs_list.append({current_output: current_data})
                    else:
                        print(
                            f"      Warning: Skipping incomplete output: '{current_output}'"
                        )

                # Start new output
                current_output = line[len("OUTPUT:") :].strip()
                current_output = current_output.replace(" ", "_").lower()
                current_data = {}

            elif line.startswith("TYPE:"):
                current_data["type"] = line[len("TYPE:") :].strip()

            elif line.startswith("DESCRIPTION:"):
                current_data["description"] = line[len("DESCRIPTION:") :].strip()

        # Don't forget to add the last output after loop ends
        if current_output and current_data:
            if "type" in current_data and "description" in current_data:
                outputs_list.append({current_output: current_data})
            else:
                print(
                    f"      Warning: Skipping incomplete output: '{current_output}'. Providing fallback values."
                )
                if current_output:
                    # Provide fallback values for incomplete outputs
                    current_data.setdefault("type", "str")  # Default type to 'str'
                    current_data.setdefault("description", "No description provided.")
                    outputs_list.append({current_output: current_data})

        if not outputs_list:
            print(f"   Warning: No outputs parsed for profile '{profile_name}'.")

    except Exception as e:
        print(f"   Error generating outputs for profile '{profile_name}': {e}")
        return []

    print(f"      Generated {len(outputs_list)} output definitions.")
    return outputs_list


def extract_conversation_paths(
    functionality_graph: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    Extract all possible conversation paths through the functionality graph,
    including single-node paths for informational chatbots.

    Args:
        functionality_graph: A structured graph of chatbot functionalities

    Returns:
        A list of paths, where each path is a list of nodes
    """
    if not functionality_graph:
        print("Warning: Empty functionality graph provided")
        return []

    # For quick node lookups
    node_lookup = {
        node.get("name"): node for node in functionality_graph if node.get("name")
    }

    # First find all possible starting points (roots)
    root_nodes = []
    for node in functionality_graph:
        # Nodes without parents are roots
        if not node.get("parents") or len(node.get("parents", [])) == 0:
            root_nodes.append(node)

    print(f"Found {len(root_nodes)} root nodes in the functionality graph")

    # If no roots found, just use all nodes (fallback)
    if not root_nodes and functionality_graph:
        print("No explicit root nodes found, using all nodes as starting points")
        root_nodes = functionality_graph

    all_paths = []

    # For each root, find all possible paths through DFS
    for root in root_nodes:
        print(f"Finding paths from root: {root.get('name', 'Unnamed')}")

        # Need these for DFS
        visited = set()
        current_path = []

        # Add single-node path (for info chatbots)
        root_path = [root]
        all_paths.append(root_path)

        # Find all paths from this root to leaf nodes
        paths_from_root = []
        find_paths_dfs(root, node_lookup, visited, current_path, paths_from_root)

        print(
            f"  Found {len(paths_from_root)} paths from {root.get('name', 'Unnamed')}"
        )
        all_paths.extend(paths_from_root)

    print(f"Total paths found: {len(all_paths)}")
    return all_paths


def find_paths_dfs(
    current_node: Dict[str, Any],
    node_lookup: Dict[str, Dict[str, Any]],
    visited: Set[str],
    current_path: List[Dict[str, Any]],
    result_paths: List[List[Dict[str, Any]]],
):
    """
    Recursively find all paths from current node to leaf nodes using DFS.

    Args:
        current_node: The current node being processed
        node_lookup: Dictionary mapping node names to node objects
        visited: Set of visited node names to prevent cycles
        current_path: List of nodes in the current path
        result_paths: List to store complete paths
    """
    node_name = current_node.get("name")

    # Skip this node if we've seen it before (avoid cycles)
    if not node_name or node_name in visited:
        return

    # Add to our current path
    visited.add(node_name)
    current_path.append(current_node)

    # Get the children of this node
    children = current_node.get("children", [])

    if not children:
        # We've hit a leaf node, save this path
        result_paths.append(list(current_path))
    else:
        # Keep going deeper for each child
        for child_info in children:
            child_name = child_info.get("name")
            if child_name and child_name in node_lookup:
                child_node = node_lookup[child_name]
                # Need to use copied collections to avoid side effects
                find_paths_dfs(
                    child_node,
                    node_lookup,
                    visited.copy(),
                    current_path.copy(),
                    result_paths,
                )

    # Clean up before returning (standard DFS backtracking)
    current_path.pop()
    visited.remove(node_name)


def generate_profile_from_path(
    path: List[Dict[str, Any]], llm, supported_languages=None
):
    """
    Generate a user profile based on a specific path through the functionality graph.

    Args:
        path: A list of nodes representing a conversation path
        llm: Language model to use for generation
        supported_languages: List of languages supported by the chatbot

    Returns:
        A dictionary containing the profile information
    """
    # Extract function names and descriptions to understand the path
    functionality_names = [node.get("name", "unnamed") for node in path]
    functionality_descriptions = []

    # Get descriptions or use the name as fallback
    for node in path:
        desc = node.get("description", "")
        if desc:
            functionality_descriptions.append(desc)
        else:
            # No description available, format the name instead
            name = node.get("name", "")
            if name:
                functionality_descriptions.append(name.replace("_", " "))

    # Set up language preference
    primary_language = (
        supported_languages[0]
        if supported_languages and supported_languages[0]
        else "English"
    )
    language_instruction = f"Write your response in {primary_language}."

    # Create prompt for profile generation
    prompt = f"""
    Create a realistic user profile for someone who would follow this conversation path with a chatbot:

    CONVERSATION PATH FUNCTIONALITIES:
    {", ".join(functionality_names)}

    FUNCTIONALITY DESCRIPTIONS:
    {". ".join(functionality_descriptions)}

    Based on these conversation functionalities, create a specific user profile with:
    1. A CONVERSATION SCENARIO NAME that summarizes the overall user need/journey
    2. A USER ROLE describing who this person is and their general motivation

    The profile should represent a realistic user who would naturally need all these functionalities in sequence.

    {language_instruction}

    FORMAT YOUR RESPONSE EXACTLY AS:
    PROFILE: [Conversation Scenario Name]
    ROLE: [Brief description of who the user is and their motivation]
    """

    # Get profile information from LLM
    response = llm.invoke(prompt)
    response_content = response.content.strip()

    # Extract profile name and role
    profile_name = None
    role = None

    for line in response_content.split("\n"):
        line = line.strip()
        if line.startswith("PROFILE:"):
            profile_name = line[len("PROFILE:") :].strip()
        elif line.startswith("ROLE:"):
            role = line[len("ROLE:") :].strip()

    # Handle parsing failure with default values
    if not profile_name or not role:
        path_name = "_".join([n.get("name", "node")[:10] for n in path[:2]])
        profile_name = f"User following path {path_name}"
        role = "Customer seeking assistance with multiple related tasks"
        print(
            "Warning: Failed to parse profile name or role from LLM response. Using fallback values."
        )

    return {"name": profile_name, "role": role, "path": functionality_names}


def generate_sequential_goals(
    path: List[Dict[str, Any]],
    llm,
    supported_languages: Optional[List[str]] = None,
    limitations: Optional[List[str]] = None,
    conversation_history: Optional[List[List[Dict[str, str]]]] = None,
) -> List[str]:
    """
    Generate a sequence of user-centric conversation goals that follows the given path.

    Args:
        path: A list of node dictionaries representing the conversation path.
        llm: The language model instance.
        supported_languages: Optional list of supported languages.
        limitations: Optional list of known chatbot limitations.
        conversation_history: Optional conversation history for context.

    Returns:
        A list of goal strings. Returns fallback goals if generation fails.
    """
    if not path:
        print("Warning: Cannot generate goals from empty path.")
        return []

    # Format information about each node in the path
    path_info_parts = []
    all_param_names_in_path = set()  # Used for suggesting variables

    for i, node in enumerate(path):
        name = node.get("name", "Unnamed Step")
        desc = node.get("description", "No description")

        # Handle parameters - they can be in different formats
        params_list = []
        raw_params = node.get("parameters", [])

        # Parameter can be a list of strings, dicts, or a single string
        if isinstance(raw_params, list):
            for p in raw_params:
                if isinstance(p, dict) and "name" in p:
                    param_name = p["name"]
                    params_list.append(param_name)
                    all_param_names_in_path.add(param_name)
                elif isinstance(p, str):
                    params_list.append(p)
                    all_param_names_in_path.add(p)
        elif isinstance(raw_params, str):
            params_list.append(raw_params)
            all_param_names_in_path.add(raw_params)

        # Build description with parameters if available
        params_str = f"(Params: {', '.join(params_list)})" if params_list else ""
        path_info_parts.append(f"  {i + 1}. {name}: {desc} {params_str}")

    # Combine all node information
    path_description = "\n".join(path_info_parts)

    # Add hint about variables if parameters exist
    possible_variables_hint = (
        f"Possible parameters to turn into {{variables}}: {', '.join(sorted(all_param_names_in_path))}"
        if all_param_names_in_path
        else ""
    )

    # Language setup
    primary_language = supported_languages[0] if supported_languages else "English"
    language_instruction = f"Write ALL goal text in {primary_language}. Keep formatting markers like 'GOALS:' in English."

    # Include limitations to avoid generating goals about them
    limitations_str = ""
    if limitations:
        limitations_str = (
            "KNOWN CHATBOT LIMITATIONS (Avoid creating goals about these):\n"
            + "\n".join(f"- {lim}" for lim in limitations)
        )

    # Add conversation examples for context
    history_str = ""
    if conversation_history:
        history_str = "EXAMPLE CONVERSATION SNIPPETS (for context):\n"
        snippet_count = 0
        for session in conversation_history:
            if snippet_count >= 2:
                break  # Just need a couple of examples
            session_text = ""
            turn_count = 0
            for turn in session:
                if turn_count >= 4:
                    break  # Keep snippets short
                role = "Human" if turn.get("role") == "assistant" else "Chatbot"
                session_text += f"{role}: {turn.get('content', '')[:80]}...\n"
                turn_count += 1
            if session_text:
                history_str += f"---\n{session_text.strip()}\n"
                snippet_count += 1
        history_str = history_str[:1500]  # Limit length

    # Determine appropriate number of goals
    num_goals_to_generate = min(5, max(2, len(path)))

    # Build the prompt
    prompt = f"""
    Generate a sequence of {num_goals_to_generate} realistic, user-centric conversation goals for a user simulator.
    These goals MUST follow this specific sequence of chatbot functionalities IN ORDER:

    Functionality Path:
    {path_description}

    {possible_variables_hint}

    {limitations_str}

    {history_str}

    {language_instruction}

    CRITICAL INSTRUCTIONS FOR GOALS:
    1.  **Strict Sequence:** Each goal MUST correspond to the next step in the Functionality Path provided above. The first goal triggers step 1, the second triggers step 2 (if applicable), and so on.
    2.  **User-Centric:** Write goals from the USER'S perspective – what they would actually SAY or ASK. Do NOT describe the functionality itself.
    3.  **Specificity:** Goals should be concrete actions or questions, not vague requests like "Get help" or "Find information".
    4.  **Variables:** Use `{{variable_name}}` placeholders ONLY for dynamic values related to the parameters listed for the corresponding step in the path. Base the `variable_name` on the parameter name (e.g., if param is `pizza_type`, use `{{pizza_type}}`). Ensure double curly braces `{{ }}`.
    5.  **Realism:** Goals should sound like a real user interacting naturally.
    6.  **Quantity:** Generate exactly {num_goals_to_generate} goals.

    Example Goal Format:
    - "I'd like to order a {{pizza_type}} pizza." (Corresponds to a step with 'pizza_type' param)
    - "What toppings are available for that?" (Follows a pizza selection step)
    - "Add a {{drink_size}} {{drink_type}} to my order." (Corresponds to a step with drink params)
    - "Proceed to checkout." (Corresponds to a checkout step)

    FORMAT YOUR RESPONSE AS A BULLETED LIST, with each goal starting with '- ':

    GOALS:
    - First specific goal...
    - Second specific goal...
    - etc.

    Respond ONLY with the 'GOALS:' heading and the bulleted list of goals.
    """

    # Call LLM and extract goals
    goals = []
    try:
        print(f"   Generating {num_goals_to_generate} goals for path...")
        response = llm.invoke(prompt)
        response_content = response.content.strip()
        print(
            f"   LLM response for goals:\n{response_content[:300]}..."
        )  # Show preview

        # Extract goals from response
        in_goals_section = False
        for line in response_content.split("\n"):
            line = line.strip()
            if line.startswith("GOALS:"):
                in_goals_section = True
                continue
            if in_goals_section and (line.startswith("- ") or line.startswith("* ")):
                goal_text = line[2:].strip().strip("\"'")  # Remove quotes if present
                if goal_text:
                    goal_text = ensure_double_curly(goal_text)  # Fix variable format
                    goals.append(goal_text)

        if not goals:
            print("   Warning: No goals parsed from LLM response.")

    except Exception as e:
        print(f"   Error during goal generation LLM call: {e}")
        goals = []

    # Create fallback goals if needed
    if not goals:
        print("   Using fallback goals based on node names.")
        fallback_goals = []
        used_params_for_fallback = set()

        # Create one goal per node (up to the limit)
        for i, node in enumerate(path):
            name = node.get("name", f"step {i + 1}")

            # Find a parameter to use as variable in the goal
            placeholder = "{{details}}"  # Default placeholder
            raw_params = node.get("parameters", [])
            if isinstance(raw_params, list):
                for p in raw_params:
                    p_name = None
                    if isinstance(p, dict) and "name" in p:
                        p_name = p["name"]
                    elif isinstance(p, str):
                        p_name = p
                    # Use first unused parameter
                    if p_name and p_name not in used_params_for_fallback:
                        placeholder = f"{{{{{p_name}}}}}"
                        used_params_for_fallback.add(p_name)
                        break
            elif (
                isinstance(raw_params, str)
                and raw_params not in used_params_for_fallback
            ):
                placeholder = f"{{{{{raw_params}}}}}"
                used_params_for_fallback.add(raw_params)

            fallback_goals.append(
                f"Ask about or perform action: {name.replace('_', ' ')} using {placeholder}"
            )
            if len(fallback_goals) >= num_goals_to_generate:
                break
        goals = fallback_goals

    print(f"   Generated goals: {goals}")
    return goals


def generate_user_profiles(
    functionality_graph: List[Dict[str, Any]],
    limitations: List[str],
    llm,
    conversation_history=None,
    supported_languages=None,
):
    """
    Generate user profiles based on paths through the functionality graph.

    Args:
        functionality_graph: A structured graph of chatbot functionalities
        limitations: Known limitations of the chatbot
        llm: Language model for generation
        conversation_history: Optional historical conversations
        supported_languages: Languages supported by the chatbot

    Returns:
        List of profile dictionaries with goals and variables
    """
    # Extract all functionalities from the graph for grouping
    all_functionalities = []
    for node in functionality_graph:
        name = node.get("name", "")
        description = node.get("description", "")
        if name:
            func_text = f"{name}: {description[:100]}..." if description else name
            all_functionalities.append(func_text)

    # Extract paths for reference (but don't create a profile for each path)
    paths = extract_conversation_paths(functionality_graph)
    print(f"Found {len(paths)} paths through the functionality graph")

    # Calculate recommended number of profiles
    num_functionalities = len(all_functionalities)
    min_profiles = 3
    max_profiles = 10
    suggested_profiles = max(min_profiles, min(max_profiles, num_functionalities))

    # Language setup
    primary_language = supported_languages[0] if supported_languages else "English"
    language_instruction_grouping = f"""
    LANGUAGE REQUIREMENT:
    - Write ALL profile names, descriptions, and functionalities in {primary_language}
    - KEEP ONLY the formatting markers (##, PROFILE:, DESCRIPTION:, FUNCTIONALITIES:) in English
    """

    # Prepare conversation history context
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = (
            "Here are some example conversations with the chatbot:\n\n"
        )
        for i, session in enumerate(conversation_history[:2], 1):  # Limit to 2 sessions
            conversation_context += f"--- SESSION {i} ---\n"
            for turn in session[:6]:  # Limit to 6 turns per session
                if turn["role"] == "assistant":  # Explorer
                    conversation_context += f"Human: {turn['content'][:150]}...\n"
                elif turn["role"] == "user":  # Chatbot's response
                    conversation_context += f"Chatbot: {turn['content'][:150]}...\n"
            conversation_context += "\n"

    # Create the grouping prompt
    grouping_prompt = f"""
    Based on these chatbot functionalities:
    {", ".join(all_functionalities)}

    {conversation_context}

    {language_instruction_grouping}

    Create {suggested_profiles} distinct user profiles, where each profile represents ONE specific conversation scenario.

    EXTREMELY IMPORTANT RESTRICTIONS:
    1. Create ONLY profiles for realistic end users with PRACTICAL GOALS
    2. NEVER create profiles about users asking about chatbot capabilities or limitations
    3. NEVER create profiles where the user is trying to test or evaluate the chatbot
    4. Focus ONLY on real-world user tasks and objectives
    5. The profiles should be genuine use cases, not meta-conversations about the chatbot itself
    6. FOCUS ON THE ACTUAL FUNCTIONALITIES PROVIDED - don't make up unrelated scenarios

    For example, if the functionalities are about ordering pizza, ALL profiles should be about pizza ordering (maybe different types of pizza orders or inquiries),
    NOT about employee onboarding, travel booking, or other unrelated topics.

    FORMAT YOUR RESPONSE AS:

    ## PROFILE: [Specific Conversation Scenario Name]
    ROLE: [Brief description of the user's role and motivation]
    FUNCTIONALITIES:
    - [functionality 1 relevant to this scenario]
    - [functionality 2 relevant to this scenario]

    ## PROFILE: [Another Specific Conversation Scenario Name]
    ... and so on
    """

    # Get scenario groupings from the LLM
    profiles_response = llm.invoke(grouping_prompt)
    profiles_content = profiles_response.content

    # Parse the profiles
    profile_sections = profiles_content.split("## PROFILE:")
    if not profile_sections[0].strip():
        profile_sections = profile_sections[1:]

    parsed_profiles = []
    for section in profile_sections:
        lines = section.strip().split("\n")
        profile_name = lines[0].strip()

        role = ""
        functionalities_list = []

        role_started = False
        func_started = False

        for line in lines[1:]:
            if line.startswith("ROLE:"):
                role_started = True
                role = line[len("ROLE:") :].strip()
                func_started = False
            elif line.startswith("FUNCTIONALITIES:"):
                role_started = False
                func_started = True
            elif func_started and line.strip().startswith("- "):
                functionalities_list.append(line.strip()[2:].strip())
            elif role_started:
                role += " " + line.strip()

        parsed_profiles.append(
            {
                "name": profile_name,
                "role": role.strip(),
                "functionalities": functionalities_list,
            }
        )

    # For each profile, find the most relevant path from our extracted paths
    final_profiles = []
    for profile in parsed_profiles:
        try:
            # Find the most relevant path for this profile's functionalities
            profile_path = find_best_path_for_profile(
                profile, paths, functionality_graph
            )

            # Generate goals based on the path and the profile
            profile["goals"] = generate_profile_goals(
                profile["name"],
                profile["role"],
                profile["functionalities"],
                profile_path,
                llm,
                supported_languages,
                limitations,
                conversation_history,
            )

            # Replace variables with placeholders for context/output generation
            sanitized_goals = [
                re.sub(VARIABLE_PATTERN, "[details]", goal)
                for goal in profile.get("goals", [])
                if isinstance(goal, str)
            ]

            # Generate additional profile components
            profile["context"] = generate_context(
                profile["name"],
                profile["role"],
                sanitized_goals,
                profile_path,
                llm,
                f"Write content in {primary_language}.",
            )

            profile["outputs"] = generate_outputs(
                profile["name"],
                profile["role"],
                sanitized_goals,
                profile_path,
                llm,
                f"Write content in {primary_language}.",
            )

            final_profiles.append(profile)
        except Exception as e:
            print(f"Error processing profile '{profile['name']}': {e}")
            continue

    # Generate definitions for all variables in the profiles
    profiles_with_vars = generate_variable_definitions(
        final_profiles, llm, supported_languages
    )

    return profiles_with_vars


def find_best_path_for_profile(profile, paths, functionality_graph):
    """Find the most relevant path for a profile based on its functionalities"""
    profile_funcs = profile.get("functionalities", [])

    # Normalize function names for comparison
    norm_profile_funcs = [func.split(":")[0].strip().lower() for func in profile_funcs]

    best_path = None
    best_score = -1

    for path in paths:
        path_func_names = [node.get("name", "").lower() for node in path]

        # Calculate match score: how many profile functions are in this path
        score = sum(
            1
            for func in norm_profile_funcs
            if any(func in path_name for path_name in path_func_names)
        )

        # Prefer longer paths if scores are equal
        if score > best_score or (
            score == best_score and len(path) > len(best_path or [])
        ):
            best_score = score
            best_path = path

    # If no good match, use the longest path available
    if not best_path:
        best_path = max(paths, key=len) if paths else []

    return best_path


def generate_profile_goals(
    profile_name,
    role,
    profile_functionalities,
    path,
    llm,
    supported_languages=None,
    limitations=None,
    conversation_history=None,
):
    """Generate goals for a profile based on its functionalities and corresponding path"""
    primary_language = supported_languages[0] if supported_languages else "English"

    language_instruction = f"""
    LANGUAGE REQUIREMENT:
    - Write ALL goals in {primary_language}
    - KEEP ONLY the formatting markers (GOALS:) in English
    - Keep variables in {{variable}} format
    """

    # Extract path information for context
    path_details = []
    for node in path:
        name = node.get("name", "")
        desc = node.get("description", "")
        params = node.get("parameters", [])
        param_str = ""
        if params:
            if isinstance(params, list):
                param_str = f" (Parameters: {', '.join(str(p) for p in params)})"
            else:
                param_str = f" (Parameter: {params})"

        if name:
            path_details.append(f"- {name}: {desc}{param_str}")

    path_info = (
        "\n".join(path_details) if path_details else "No specific path available"
    )

    # Format conversation history snippet if available
    conversation_snippet = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_snippet = "EXAMPLE CONVERSATION:\n"
        session = conversation_history[0][:6]  # First session, up to 6 turns
        for turn in session:
            role = "Human" if turn.get("role") == "assistant" else "Chatbot"
            conversation_snippet += f"{role}: {turn.get('content', '')[:100]}...\n"

    # Create the goals prompt
    goals_prompt = f"""
    Generate a set of coherent user-centric goals for this conversation scenario:

    CONVERSATION SCENARIO: {profile_name}
    USER ROLE: {role}

    RELEVANT FUNCTIONALITIES:
    {", ".join(profile_functionalities)}

    CONVERSATION PATH DETAILS:
    {path_info}

    {conversation_snippet}

    {language_instruction}

    ABOUT VARIABLES:
    - Use {{variable_name}} where the user might provide different values (e.g., {{pizza_type}}, {{size}})
    - Variables should be based on parameters identified in the conversation path
    - Use double curly braces for variables (e.g., {{variable}})

    EXTREMELY IMPORTANT REQUIREMENTS:
    1. Create 2-4 goals that form a NATURAL CONVERSATION FLOW
    2. Goals must be strictly about {profile_name} - not about testing or evaluating the chatbot
    3. Make goals specific to what a user would say or ask, not what the chatbot would do
    4. Focus on practical, realistic user requests/questions
    5. The goals should follow a logical sequence matching the conversation path

    FORMAT YOUR RESPONSE AS:

    GOALS:
    - "First specific user goal with {{variable}} if needed"
    - "Second related goal that progresses the conversation"
    - "Third goal that follows logically"

    Make goals sound like natural user utterances, not instructions or descriptions.
    """

    # Get goals from LLM
    goals_response = llm.invoke(goals_prompt)
    goals_content = goals_response.content

    # Parse goals
    goals = []
    if "GOALS:" in goals_content:
        goals_section = goals_content.split("GOALS:")[1].strip()
        for line in goals_section.split("\n"):
            if line.strip().startswith("- "):
                goal = line.strip()[2:].strip().strip("\"'")
                if goal:
                    goal = ensure_double_curly(goal)
                    goals.append(goal)

    # If no goals parsed, create fallback goals
    if not goals:
        print(
            f"Warning: No goals parsed for profile '{profile_name}'. Using fallback goals."
        )
        function_count = min(4, len(profile_functionalities))
        for i in range(function_count):
            if i < len(profile_functionalities):
                func_text = profile_functionalities[i]
                func_name = (
                    func_text.split(":")[0].strip() if ":" in func_text else func_text
                )
                goals.append(
                    f"I need help with {func_name.replace('_', ' ')} for {{{{parameter}}}}"
                )

    return goals
