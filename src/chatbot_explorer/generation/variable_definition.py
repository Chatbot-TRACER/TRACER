from typing import Any

from chatbot_explorer.constants import VARIABLE_PATTERN

# --- Variable Definition Logic ---


def _build_single_variable_prompt(
    profile_name: str,
    role: str,
    goals_text: str,
    variable_name: str,
    all_other_variables: list[str],
    language_instruction: str,
) -> str:
    """Creates the specific text prompt to ask the LLM about one variable.

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
        f"OTHER VARIABLES IN THIS PROFILE: {', '.join(sorted(all_other_variables))}" if all_other_variables else ""
    )

    return f"""
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


def _parse_single_variable_definition(
    response_content: str,
    expected_type: str | None = None,
) -> dict[str, Any] | None:
    """Takes the LLM's text answer for one variable and tries to turn it into a Python dictionary.

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
                    f"Warning: LLM returned type '{parsed_type}' but expected '{expected_type}'. Will attempt to parse as '{parsed_type}'.",
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

    if not definition.get("function") or not definition.get("type") or "data" not in definition:
        print(
            f"Warning: Failed to parse essential fields (FUNCTION, TYPE, DATA) from LLM response:\n{response_content}",
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
            print(f"Warning: String variable data is empty. LLM response:\n{response_content}")
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
                    print(f"Warning: Could not parse numeric value for key '{key}': '{value_str}'. Skipping.")
        # Validate numeric data structure
        if not ("min" in raw_data and "max" in raw_data and ("step" in raw_data or "linspace" in raw_data)):
            print(
                f"Warning: Numeric variable data missing min/max/step or min/max/linspace. LLM response:\n{response_content}",
            )
            # Allow partial definition for potential fixing later? Or return None? Let's return None for now.
            return None
        definition["data"] = raw_data
    else:
        print(f"Warning: Unknown variable type '{data_type}'. Cannot parse data correctly.")
        return None  # Unknown type

    # Basic validation
    if not definition.get("function") or not definition.get("type") or not definition.get("data"):
        print(f"Warning: Post-parsing validation failed. Missing fields or empty data. Parsed: {definition}")
        return None

    return definition


def generate_variable_definitions(profiles, llm, supported_languages=None, max_retries=3):
    """Generate the {{variable}} definitions.

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
        language_instruction = f"Generate examples/values in {primary_language} where appropriate."

    # Process each profile
    for profile in profiles:
        # Extract all unique variables from the goals
        all_variables: set[str] = set()
        goals_list = profile.get("goals", [])
        if not isinstance(goals_list, list):  # Ensure goals is a list
            print(
                f"Warning: Profile '{profile.get('name', 'Unnamed')}' has invalid 'goals' format. Skipping variable definition.",
            )
            continue

        for goal in goals_list:
            if isinstance(goal, str):
                variables_in_goal = VARIABLE_PATTERN.findall(goal)
                all_variables.update(variables_in_goal)
            # Ignore non-string goals (like existing variable definitions if run multiple times)

        if not all_variables:
            print(f"Info: No variables found in goals for profile '{profile.get('name', 'Unnamed')}'.")
            continue

        print(f"\n--- Defining variables for profile: {profile.get('name', 'Unnamed')} ---")
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
                        print(f"      Successfully parsed definition for '{variable_name}'.")
                        break  # Success
                    print(
                        f"      Warning: Failed to parse definition for '{variable_name}' on attempt {attempt + 1}. Retrying...",
                    )

                except Exception as e:
                    print(
                        f"      Error during LLM call or parsing for '{variable_name}' on attempt {attempt + 1}: {e}. Retrying...",
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
                    goals_list.append({variable_name: parsed_def})  # Append new definition dict

            else:
                print(
                    f"      ERROR: Failed to generate valid definition for variable '{variable_name}' after {max_retries} attempts.",
                )

        # Update the profile's goals list with the added/updated definitions
        profile["goals"] = goals_list

    return profiles
