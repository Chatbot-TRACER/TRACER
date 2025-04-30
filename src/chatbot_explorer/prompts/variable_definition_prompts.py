"""Prompts for defining variables in user simulator profiles."""

def get_variable_definition_prompt(
    profile_name: str,
    role: str,
    goals_text: str,
    variable_name: str,
    all_other_variables: list[str],
    language_instruction: str,
) -> str:
    """Creates the specific text prompt to ask the LLM about one variable."""
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
