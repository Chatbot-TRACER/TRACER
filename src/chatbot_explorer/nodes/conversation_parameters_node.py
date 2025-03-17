"""Node for generating conversation parameters."""


def generate_conversation_parameters(
    profiles, functionalities, llm, supported_languages=None
):
    """Generate conversation parameters for test profiles."""

    for profile in profiles:
        # Identify variables and their relationships
        variables = []
        forward_with_dependencies = []

        for var_name, var_def in profile.items():
            if (
                isinstance(var_def, dict)
                and "function" in var_def
                and "data" in var_def
            ):
                variables.append(var_name)

                # Identify forward variables with dependencies
                if (
                    "forward" in var_def["function"]
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
                variables_info += f"\n{len(forward_with_dependencies)} variables have nested dependencies: {', '.join(forward_with_dependencies)}"
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

        conversation_params_prompt = f"""
        Generate appropriate conversation parameters for this user profile.

        CONVERSATION SCENARIO: {profile["name"]}
        USER ROLE: {profile["role"]}
        {variables_info}{language_info}

        PARAMETERS TO DETERMINE:

        1. NUMBER:
           Choose ONE option based on the variables detected:
           - A specific number (2-5) if there are no variables or no nested dependencies
           - "all_combinations" if there are variables but no nested dependencies AND you want to test ALL combinations
           - "sample(X)" where X is a decimal between 0.1 and 0.9 for testing a fraction of the combinations

           RECOMMENDATION:
           - If no variables: use 2-5 conversations (fewer for simple tasks, more for complex ones)
           - If variables with NO nested dependencies: use all_combinations if there are less than 5 combinations, otherwise use sample(X)
           - If variables WITH nested dependencies: NEVER use "all_combinations", only use "sample(X)" with X less than 0.5

        2. MAX_COST:
           - Set a budget limit for all conversations combined, in dollars
           - For specific number conversations: typically 0.5-1.0 dollars
           - For "all_combinations": use higher limits (1.5-3.0) since there will be more conversations
           - For "sample(X)": scale based on X value - higher X needs higher budget
           - IMPORTANT: Consider the complexity of the conversation and goals when setting this limit

        3. GOAL_STYLE:
           Choose ONE option that best fits this conversation scenario:
           - "steps": Fixed number of conversation turns between 5-12, use for simple conversations with predictable flow
           - "all_answered": Ends when all user goals are addressed (can include "limit" parameters so the conversation length is still capped), use when testing the goals are very different from one another so it will be difficult to reach them or when testing something that looks like a core feature.
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

        params_response = llm.invoke(conversation_params_prompt)
        response_text = params_response.content.strip()

        # Set defaults based on variable types
        if forward_with_dependencies:
            default_number = "sample(0.3)"
        elif variables:
            default_number = 3 if len(variables) <= 5 else "sample(0.5)"
        else:
            default_number = 2

        # Initialize parameters
        conversation_params = {
            "number": default_number,
            "max_cost": 1.0,
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
                if value == "all_combinations":
                    extracted_number = "all_combinations"
                elif "sample" in value.lower() and "(" in value and ")" in value:
                    try:
                        sample_value = float(value.split("(")[1].split(")")[0])
                        if 0 < sample_value <= 1:
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

        # Validate NUMBER parameter
        if extracted_number is not None:
            if (
                not variables
                and isinstance(extracted_number, str)
                and "sample" in extracted_number
            ):
                # Can't use sample without variables
                pass  # Keep default_number
            elif len(variables) > 10 and extracted_number == "all_combinations":
                # Too many variables for all_combinations
                conversation_params["number"] = "sample(0.3)"
            else:
                conversation_params["number"] = extracted_number

        # Add interaction styles
        if len(interaction_styles) == 1:
            conversation_params["interaction_style"] = interaction_styles[0]
        elif len(interaction_styles) > 1:
            conversation_params["interaction_style"] = interaction_styles

        profile["conversation"] = conversation_params

    return profiles
