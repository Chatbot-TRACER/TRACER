"""Node for generating conversation parameters."""


def generate_conversation_parameters(
    profiles, functionalities, llm, supported_languages=None
):
    """Generate conversation parameters for test profiles."""

    for profile in profiles:
        # Identify variables and their relationships
        variables = []
        forward_variables = []
        forward_with_dependencies = []

        for var_name, var_def in profile.items():
            if (
                isinstance(var_def, dict)
                and "function" in var_def
                and "data" in var_def
            ):
                variables.append(var_name)

                if "forward" in var_def["function"]:
                    forward_variables.append(var_name)

                    if "(" in var_def["function"] and ")" in var_def["function"]:
                        param = var_def["function"].split("(")[1].split(")")[0]
                        if param and param != "rand" and not param.isdigit():
                            forward_with_dependencies.append(var_name)

        # Build profile information for the prompt
        variables_info = ""
        if variables:
            variables_info = (
                f"\nThis profile has {len(variables)} variables: {', '.join(variables)}"
            )

            if forward_variables:
                variables_info += f"\n{len(forward_variables)} of these use forward functions: {', '.join(forward_variables)}"

                if forward_with_dependencies:
                    variables_info += f"\n{len(forward_with_dependencies)} forward variables have dependencies: {', '.join(forward_with_dependencies)}"
                    variables_info += "\nThis creates COMBINATIONS that could be explored with 'all_combinations' or 'sample(X)'."

        # Prepare language information
        language_info = ""
        languages_example = ""
        supported_languages_text = ""

        if supported_languages and len(supported_languages) > 0:
            language_info = f"\nSUPPORTED LANGUAGES: {', '.join(supported_languages)}"
            supported_languages_text = f"({', '.join(supported_languages)})"

            language_lines = []
            for lang in supported_languages:
                language_lines.append(f"- {lang.lower()}")
            languages_example = "\n".join(language_lines)

        # Generate prompt for the LLM to determine conversation parameters
        conversation_params_prompt = f"""
        Generate appropriate conversation parameters for this user profile.

        CONVERSATION SCENARIO: {profile["name"]}
        USER ROLE: {profile["role"]}
        {variables_info}{language_info}

        PARAMETERS TO DETERMINE:

        1. NUMBER:
           Choose ONE option based on the variables detected:
           - A specific number (2-5) if there are no forward functions with dependencies
           - "all_combinations" if there are forward variables with dependencies AND you want to test ALL combinations
           - "sample(X)" where X is a decimal between 0.1 and 0.9 for testing a fraction of the combinations

           RECOMMENDATION:
           - If no variables: use 2-3 conversations
           - If variables with NO dependencies: use all_combinations if there are less than 5 combinations, if not use a sample(X), if each of the variables is very different from each other use a higher X, if they are more similar use a lower one since it won't make sense to test a lot of similar ones.
           - If forward variables WITH dependencies: dont use "all_combinations" since that would be too much, use "sample(X)" as was explained before.

        2. MAX_COST:
           - Set a budget limit for all conversations combined, in dollars
           - For specific number conversations: typically 0.5-1.0 dollars
           - For "all_combinations": use higher limits (1.5-3.0) since there will be more conversations
           - For "sample(X)": scale based on X value - higher X needs higher budget
           - IMPORTANT: Consider the complexity of the conversation and goals when setting this limit

        3. GOAL_STYLE:
           Choose ONE option that best fits this conversation scenario:
           - "steps": Fixed number of conversation turns between 5-12, use for simple conversations with predictable flow
           - "all_answered": Ends when all user goals are addressed (can include "limit" parameters so the conversation length is still capped), use when testing if multiple user requirements are fulfilled
           - "random_steps": Random number of turns up to a maximum value, use when testing chatbot resilience with varied interaction lengths

           EXAMPLE FORMAT:
           - For steps: {{"steps": 7}}
           - For all_answered: {{"all_answered": {{"limit": 20}}}}
           - For random_steps: {{"random_steps": 12}}


        4. INTERACTION_STYLE:
           Choose 1-2 appropriate styles based on the user role and scenario:
           - "long phrases": user uses very long phrases to write any query
           - "change your mind": user changes their mind during conversation (good for ordering/booking)
           - "change language": user switches languages mid-conversation (requires list of languages)
           - "make spelling mistakes": user makes typos and spelling mistakes
           - "single question": user asks only one query per interaction
           - "all questions": user asks everything from goals in one interaction
           - "random": applies random styles from a specified list
           - If no style is specified, "default" is used (natural conversation style)

           RECOMMENDATION:
           - Select styles that match your user's likely behavior
           - Use "change your mind" for scenarios where selections are made
           - Use "single question" for complex conversations, "all questions" for simpler ones
           - Use "random" with a list when you want varied interaction styles
           - Use "change language" when the chatbot supports multiple languages {supported_languages_text}


           EXAMPLES:
           - Single style: "make spelling mistakes"
           - Multiple styles: ["long phrases", "change your mind"]
           - Random with language change:
             random:
               - make spelling mistakes
               - all questions
               - long phrases
               - change language:
                {languages_example}

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        NUMBER: specific_number or all_combinations or sample(0.X)
        MAX_COST: number
        GOAL_STYLE: type
        GOAL_STYLE_VALUE: value
        INTERACTION_STYLE: style or [style1, style2]
        """

        # Get LLM response
        params_response = llm.invoke(conversation_params_prompt)
        response_text = params_response.content.strip()

        # Set default parameters based on variable types
        if forward_with_dependencies:
            default_number = "all_combinations"
        elif forward_variables:
            default_number = 3
        elif variables:
            default_number = 2
        else:
            default_number = 2

        # Initialize conversation parameters with defaults
        # These will be updated based on the LLM response
        conversation_params = {
            "number": default_number,
            "max_cost": 1.0,
            "goal_style": {"steps": 10},
        }

        # Parse LLM response to extract parameters
        interaction_styles = []
        for line in response_text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            # Try to parse the number value
            if key == "number":
                if value == "all_combinations":
                    conversation_params["number"] = "all_combinations"
                elif "sample" in value.lower() and "(" in value and ")" in value:
                    try:
                        sample_value = float(value.split("(")[1].split(")")[0])
                        if 0 < sample_value <= 1:
                            conversation_params["number"] = f"sample({sample_value})"
                    except ValueError:
                        if forward_with_dependencies:
                            conversation_params["number"] = "sample(0.2)"
                elif value.isdigit():
                    conversation_params["number"] = int(value)

            elif key == "max_cost":
                try:
                    conversation_params["max_cost"] = float(value)
                except ValueError:
                    pass

            elif key == "goal_style":
                if value in ["steps", "all_answered", "random_steps"]:
                    if value == "steps":
                        conversation_params["goal_style"] = {"steps": 10}
                    elif value == "all_answered":
                        conversation_params["goal_style"] = {
                            "all_answered": {"export": False, "limit": 30}
                        }
                    elif value == "random_steps":
                        conversation_params["goal_style"] = {"random_steps": 15}

            elif key == "goal_style_value":
                if "steps" in conversation_params["goal_style"]:
                    try:
                        conversation_params["goal_style"] = {"steps": int(value)}
                    except ValueError:
                        pass
                elif "random_steps" in conversation_params["goal_style"]:
                    try:
                        conversation_params["goal_style"] = {"random_steps": int(value)}
                    except ValueError:
                        pass

            elif key == "interaction_style":
                if "[" in value and "]" in value:
                    styles_part = value.replace("[", "").replace("]", "")
                    styles = [s.strip().strip("\"'") for s in styles_part.split(",")]
                    interaction_styles = [s for s in styles if s]
                else:
                    style = value.strip().strip("\"'")
                    if style:
                        interaction_styles.append(style)

        # Add interaction styles to parameters
        if len(interaction_styles) == 1:
            conversation_params["interaction_style"] = interaction_styles[0]
        elif len(interaction_styles) > 1:
            conversation_params["interaction_style"] = interaction_styles

        # Store parameters in profile
        profile["conversation"] = conversation_params

    return profiles
