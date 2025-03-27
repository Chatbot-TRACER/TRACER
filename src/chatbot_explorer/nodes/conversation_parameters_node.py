"""Node for generating conversation parameters."""


def generate_conversation_parameters(
    profiles, functionalities, llm, supported_languages=None
):
    """Generate conversation parameters for test profiles."""

    for profile in profiles:
        # Identify variables and their relationships
        variables = []
        forward_with_dependencies = []
        has_nested_forwards = False
        nested_forward_info = ""

        # Get all variables regardless of type
        for var_name, var_def in profile.items():
            if (
                isinstance(var_def, dict)
                and "function" in var_def
                and "data" in var_def
            ):
                variables.append(var_name)

        # Use the computed forward dependency information from goals_node if available
        if "has_nested_forwards" in profile:
            has_nested_forwards = profile["has_nested_forwards"]

            if "forward_dependencies" in profile:
                forward_dependencies = profile["forward_dependencies"]
                forward_with_dependencies = list(forward_dependencies.keys())

                # Create detailed nested forward information
                if has_nested_forwards and "nested_forward_chains" in profile:
                    nested_chains = profile["nested_forward_chains"]
                    chain_descriptions = []

                    for chain in nested_chains:
                        chain_str = " → ".join(chain)
                        chain_descriptions.append(f"Chain: {chain_str}")

                    if chain_descriptions:
                        nested_forward_info = (
                            "\nNested dependency chains detected:\n"
                            + "\n".join(chain_descriptions)
                        )

                        # Calculate potential combinations if possible
                        combinations = 1
                        for var_name in variables:
                            var_def = profile.get(var_name, {})
                            if isinstance(var_def, dict) and "data" in var_def:
                                data = var_def.get("data", [])
                                if isinstance(data, list):
                                    combinations *= len(data)
                                elif (
                                    isinstance(data, dict)
                                    and "min" in data
                                    and "max" in data
                                    and "step" in data
                                ):
                                    steps = (data["max"] - data["min"]) / data[
                                        "step"
                                    ] + 1
                                    combinations *= int(steps)

                        nested_forward_info += (
                            f"\nPotential combinations: approximately {combinations}"
                        )
            else:
                # Fallback to current detection method if forward_dependencies isn't available
                for var_name, var_def in profile.items():
                    if (
                        isinstance(var_def, dict)
                        and "function" in var_def
                        and "data" in var_def
                        and "forward" in var_def["function"]
                        and "(" in var_def["function"]
                        and ")" in var_def["function"]
                    ):
                        param = var_def["function"].split("(")[1].split(")")[0]
                        if param and param != "rand" and not param.isdigit():
                            forward_with_dependencies.append(var_name)

        # Build profile information for the prompt
        variables_info = ""
        if variables:
            variables_info = (
                f"\nThis profile has {len(variables)} variables: {', '.join(variables)}"
            )

            if forward_with_dependencies:
                variables_info += f"\n{len(forward_with_dependencies)} variables have dependencies: {', '.join(forward_with_dependencies)}"
                if has_nested_forwards:
                    variables_info += "\nThis creates COMBINATIONS that could be explored with 'all_combinations', 'sample(X)', or a fixed number."
                    variables_info += f"\nIMPORTANT: This profile has NESTED FORWARD DEPENDENCIES.{nested_forward_info}"

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

        # Select the appropriate prompt based on whether nested forwards exist
        if has_nested_forwards:
            # Prompt for profiles WITH nested forward dependencies
            number_section = """
            1. NUMBER:
               Choose ONE option:
               - Enter a fixed number (e.g., 2-5) for a specific number of conversations.
               - "all_combinations" to try all possible combinations (ONLY if there are fewer than 5 total combinations).
               - "sample(X)" where X is a decimal between 0.1 and 1.0 to test a fraction of the combinations.

               RECOMMENDATION:
               - For nested forwards, if total combinations are low (<5), you may choose "all_combinations".
               - Otherwise, consider using "sample(0.2)" or "sample(0.5)" based on your testing needs, or simply specify a fixed number.
            """
            default_number = "sample(0.2)"
        else:
            # Prompt for profiles WITHOUT nested forward dependencies
            number_section = """
            1. NUMBER:
               Choose a specific number between 2-5 conversations to generate.

               RECOMMENDATION:
               - 2-3 for simple, straightforward conversations.
               - 4-5 for more complex scenarios with multiple user goals.
            """
            default_number = 5 if variables else 2

        conversation_params_prompt = f"""
        Generate appropriate conversation parameters for this user profile.

        CONVERSATION SCENARIO: {profile["name"]}
        USER ROLE: {profile["role"]}
        {variables_info}{language_info}

        PARAMETERS TO DETERMINE:

        {number_section}

        2. MAX_COST:
           - Set a budget limit for all conversations combined, in dollars.
           - For specific number conversations: typically 0.5-1.0 dollars.
           - For "all_combinations": use higher limits (1.5-3.0) since there will be more conversations.
           - For "sample(X)": scale based on X value - higher X needs higher budget.
           - IMPORTANT: Consider the complexity of the conversation and goals when setting this limit.

        3. GOAL_STYLE:
           Choose ONE option that best fits this conversation scenario:
           - "steps": Fixed number of conversation turns between 5-12, use for simple conversations with predictable flow.
           - "all_answered": Ends when all user goals are addressed (can include "limit" parameters so the conversation length is still capped), use when testing the goals are very different from one another so it will be difficult to reach them or when testing something that looks like a core feature. Set a limit of 10 to 20 steps.
           - "random_steps": Random number of turns up to a maximum value, use when testing chatbot resilience with varied interaction lengths.

           EXAMPLE FORMAT:
           - For steps: {{"steps": 7}}
           - For all_answered: {{"all_answered": {{"limit": 10}}}}
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

        params_response = llm.invoke(conversation_params_prompt)
        response_text = params_response.content.strip()

        # Initialize parameters with profile-specific defaults
        conversation_params = {
            "number": default_number,
            "max_cost": 1.0 if not has_nested_forwards else 2.0,
            "goal_style": {"steps": 10},
        }
        interaction_styles = []
        extracted_number = None

        for line in response_text.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "number":
                # For nested forwards allow fixed number, all_combinations, or sample(X)
                if value == "all_combinations":
                    extracted_number = "all_combinations"
                elif "sample" in value.lower() and "(" in value and ")" in value:
                    try:
                        sample_value = float(value.split("(")[1].split(")")[0])
                        if 0.1 <= sample_value <= 1.0:
                            extracted_number = f"sample({sample_value})"
                    except ValueError:
                        pass
                elif value.isdigit():
                    extracted_number = int(value)
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

        if extracted_number is not None:
            conversation_params["number"] = extracted_number
        if interaction_styles:
            conversation_params["interaction_style"] = (
                interaction_styles[0]
                if len(interaction_styles) == 1
                else interaction_styles
            )

        profile["conversation"] = conversation_params

    return profiles
