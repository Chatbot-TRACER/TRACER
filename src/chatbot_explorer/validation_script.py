from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import yaml
import re


@dataclass
class ValidationError:
    message: str
    path: str
    line: Optional[int] = None


class YamlValidator:
    def __init__(self):
        self.required_top_level = [
            "test_name",
            # "llm",
            "user",
            "conversation",
            "chatbot",
        ]
        # These are mandatory, if they are not present, the validation will fail
        self.required_nested = {
            # "llm": ["model", "temperature", "format"],
            "llm.format": ["type"],
            "user": ["role", "context", "goals"],
            "chatbot": ["is_starter", "fallback", "output"],
            "conversation": ["number", "goal_style", "interaction_style"],
        }
        # While these are not mandatory, but they are only allowed keywords
        self.allowed_fields = {
            "llm": ["model", "temperature", "format"],
        }
        self.valid_formats = ["text", "speech"]
        self.valid_goal_functions = ["default()", "random()", "another()", "forward()"]
        self.valid_variable_types = ["string", "float", "int"]
        self.valid_output_types = [
            "int",
            "float",
            "money",
            "str",
            "string",
            "time",
            "date",
        ]
        self.valid_goal_styles = [
            "steps",
            "random_steps",
            "all_answered",
            "max_cost",
            "default",
        ]
        self.valid_interaction_styles = [
            "long phrases",
            "change your mind",
            "make spelling mistakes",
            "single question",
            "all questions",
            "default",
        ]

    def validate(self, yaml_content: str) -> List[ValidationError]:
        """Validate YAML content against schema rules.
        Checks that all required fields are present and have the correct type.
        """
        errors = []

        try:
            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                return [ValidationError("Root level must be a dictionary", "/")]

            # Validate top-level required fields
            errors.extend(self._validate_required_fields(data))

            # Validate specific sections
            if "llm" in data:
                errors.extend(self._validate_llm_section(data["llm"]))

            if "user" in data:
                errors.extend(self._validate_user_section(data["user"]))

            if "chatbot" in data:
                errors.extend(self._validate_chatbot_section(data["chatbot"]))

            if "conversation" in data:
                errors.extend(self._validate_conversation_section(data["conversation"]))

            if "user" in data and "conversation" in data:
                errors.extend(
                    self._validate_conversation_variable_dependencies(
                        data["user"], data["conversation"]
                    )
                )

            return errors

        except yaml.YAMLError as e:
            return [ValidationError(f"Invalid YAML syntax: {str(e)}", "/")]

    def _validate_conversation_variable_dependencies(
        self, user: Dict, conversation: Dict
    ) -> List[ValidationError]:
        """Validate that sample() or all_combinations is only used when there are nested forwards."""
        errors = []

        # Check if conversation.number is using sample() or all_combinations
        using_combinations = False
        if "number" in conversation:
            num = conversation["number"]
            if isinstance(num, str):
                if num == "all_combinations" or (
                    isinstance(num, str) and num.startswith("sample(")
                ):
                    using_combinations = True

        # If using combinations, check for nested forwards
        if using_combinations:
            has_nested_forwards = False

            # Find all forwards in the user goals
            if "goals" in user and isinstance(user["goals"], list):
                for goal in user["goals"]:
                    if isinstance(goal, dict) and len(goal) == 1:
                        var_name = list(goal.keys())[0]
                        var_def = goal[var_name]

                        if isinstance(var_def, dict) and "function" in var_def:
                            func = var_def["function"]
                            # Check if this is a forward function referring to another variable
                            if (
                                func.startswith("forward(")
                                and func.endswith(")")
                                and func != "forward()"
                            ):
                                has_nested_forwards = True
                                break

            # If using combinations but no nested forwards, show an error
            if not has_nested_forwards:
                errors.append(
                    ValidationError(
                        "Using 'all_combinations' or 'sample()' requires at least one variable with nested forward() dependency",
                        "/conversation/number",
                    )
                )

        return errors

    def _validate_required_fields(self, data: Dict) -> List[ValidationError]:
        """Validate that all required fields are present."""
        errors = []

        for field in self.required_top_level:
            if field not in data:
                errors.append(
                    ValidationError(f"Missing required field: {field}", f"/{field}")
                )

        # Check nested required fields
        for path, fields in self.required_nested.items():
            parts = path.split(".")
            current = data
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    break
                current = current[part]
            else:
                for field in fields:
                    if not isinstance(current, dict) or field not in current:
                        errors.append(
                            ValidationError(
                                f"Missing required field: {field} in {path}",
                                f"/{path}/{field}",
                            )
                        )

        return errors

    def _validate_llm_section(self, llm: Dict) -> List[ValidationError]:
        """Validate LLM section configuration."""
        errors = []

        # Check for unexpected fields in the llm section
        for field in llm:
            if field not in self.allowed_fields["llm"]:
                errors.append(
                    ValidationError(
                        f"Unexpected field '{field}' in llm section. Did you mean one of: {', '.join(self.allowed_fields['llm'])}?",
                        f"/llm/{field}",
                    )
                )

        # Check that temperature is a number between 0 and 1
        if "temperature" in llm:
            temp = llm["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 1:
                errors.append(
                    ValidationError(
                        "Temperature must be a number between 0 and 1",
                        "/llm/temperature",
                    )
                )

        # Validate format section
        if "format" in llm and isinstance(llm["format"], dict):
            format_section = llm["format"]
            format_type = format_section.get("type")

            # Validate format type
            if format_type and format_type not in self.valid_formats:
                errors.append(
                    ValidationError(
                        f"Invalid format type. Must be one of: {', '.join(self.valid_formats)}",
                        "/llm/format/type",
                    )
                )

            # If type is speech, check for config
            if format_type == "speech":
                if "config" not in format_section:
                    errors.append(
                        ValidationError(
                            "Speech format requires 'config' field with path to configuration file",
                            "/llm/format/config",
                        )
                    )
                # Check that the string is a path
                elif format_section["config"] == "":
                    errors.append(
                        ValidationError(
                            "Speech format requires 'config' field with path to configuration file",
                            "/llm/format/config",
                        )
                    )

        return errors

    def _validate_user_section(self, user: Dict) -> List[ValidationError]:
        """Validate user section configuration."""
        errors = []

        # Validate context structure
        if "context" in user:
            if not isinstance(user["context"], list):
                errors.append(
                    ValidationError(
                        "User context must be a list",
                        "/user/context",
                    )
                )
            else:
                # Track if we've seen a personality item
                has_personality = False

                for i, item in enumerate(user["context"]):
                    # Check if it's a personality item (dictionary with personality key)
                    if isinstance(item, dict):
                        if "personality" in item:
                            if has_personality:
                                errors.append(
                                    ValidationError(
                                        "Only one personality entry is allowed in context",
                                        f"/user/context/{i}",
                                    )
                                )
                            has_personality = True

                            # Validate that personality points to a path
                            if (
                                not isinstance(item["personality"], str)
                                or item["personality"] == ""
                            ):
                                errors.append(
                                    ValidationError(
                                        "Personality must be a non-empty string path",
                                        f"/user/context/{i}/personality",
                                    )
                                )
                    # All non-dictionary items must be strings
                    elif not isinstance(item, str):
                        errors.append(
                            ValidationError(
                                "Context items must be either a personality dictionary or string",
                                f"/user/context/{i}",
                            )
                        )

        # Validate goals structure
        if "goals" in user:
            if not isinstance(user["goals"], list):
                errors.append(
                    ValidationError(
                        "User goals must be a list",
                        "/user/goals",
                    )
                )
            else:
                # First pass: collect all defined variables and their functions
                defined_variables = {}
                for i, goal in enumerate(user["goals"]):
                    if isinstance(goal, dict) and len(goal) == 1:
                        var_name = list(goal.keys())[0]
                        var_def = goal[var_name]

                        if isinstance(var_def, dict) and "function" in var_def:
                            defined_variables[var_name] = {
                                "function": var_def["function"],
                                "index": i,
                            }
                        else:
                            defined_variables[var_name] = {"function": None, "index": i}

                # Second pass: validate all goals (strings and variable definitions)
                for i, goal in enumerate(user["goals"]):
                    # Goals can be either strings (prompts) or dictionaries (variables)
                    if isinstance(goal, str):
                        # First find and validate all properly formatted variables
                        # This regex now allows variable placeholders with spaces, e.g., {{asunto específico}}
                        valid_vars = re.findall(r"{{\s*([^{}]+?)\s*}}", goal)

                        # Then check if there are any other curly braces in the string
                        # Remove all valid variable placeholders
                        clean_goal = re.sub(r"{{\s*[^{}]+?\s*}}", "", goal)

                        # Now if there are any remaining curly braces, they are invalid
                        if "{" in clean_goal or "}" in clean_goal:
                            # Find the first instance of the remaining invalid brace pattern
                            invalid_pattern = re.search(r"[^{]*[{}]+[^}]*", clean_goal)
                            if invalid_pattern:
                                pattern_text = invalid_pattern.group(0).strip()
                                errors.append(
                                    ValidationError(
                                        f"Invalid use of curly braces: '{pattern_text}'. "
                                        f"Curly braces can only be used for variable placeholders with exactly two opening and two closing braces, e.g., '{{ variable }}'.",
                                        f"/user/goals/{i}",
                                    )
                                )

                        # Check if properly formatted variables are defined (as before)
                        for var in valid_vars:
                            if var not in defined_variables:
                                errors.append(
                                    ValidationError(
                                        f"Variable '{var}' used in goal but not defined",
                                        f"/user/goals/{i}",
                                    )
                                )

                        # Now if there are any remaining curly braces, they are invalid
                        if "{" in clean_goal or "}" in clean_goal:
                            # Find the first instance of the remaining invalid brace pattern
                            invalid_pattern = re.search(r"[^{]*[{}]+[^}]*", clean_goal)
                            if invalid_pattern:
                                pattern_text = invalid_pattern.group(0).strip()
                                errors.append(
                                    ValidationError(
                                        f"Invalid use of curly braces: '{pattern_text}'. "
                                        f"Curly braces can only be used for variables with exactly two opening and two closing braces: '{{{{variable}}}}'. "
                                        f"Variables must be single words without spaces.",
                                        f"/user/goals/{i}",
                                    )
                                )

                        # Check if properly formatted variables are defined (as before)
                        for var in valid_vars:
                            if var not in defined_variables:
                                errors.append(
                                    ValidationError(
                                        f"Variable '{var}' used in goal but not defined",
                                        f"/user/goals/{i}",
                                    )
                                )

                        # Then check properly formatted variables are defined
                        var_placeholders = re.findall(r"{{(\w+)}}", goal)
                        for var in var_placeholders:
                            # Make sure variables used are defined somewhere in the goals
                            if var not in defined_variables:
                                errors.append(
                                    ValidationError(
                                        f"Variable '{var}' used in goal but not defined",
                                        f"/user/goals/{i}",
                                    )
                                )

                    elif isinstance(goal, dict):
                        # Each dict should have only one key, which is the variable name
                        if len(goal) != 1:
                            errors.append(
                                ValidationError(
                                    "Each variable definition must have exactly one key",
                                    f"/user/goals/{i}",
                                )
                            )
                        else:
                            var_name = list(goal.keys())[0]
                            var_def = goal[var_name]

                            # Check required fields for variable definition
                            if not isinstance(var_def, dict):
                                errors.append(
                                    ValidationError(
                                        f"Variable '{var_name}' definition must be a dictionary",
                                        f"/user/goals/{i}/{var_name}",
                                    )
                                )
                                continue

                            # Check required fields
                            for field in ["function", "type", "data"]:
                                if field not in var_def:
                                    errors.append(
                                        ValidationError(
                                            f"Missing required field '{field}' in variable '{var_name}'",
                                            f"/user/goals/{i}/{var_name}",
                                        )
                                    )

                            # Validate type
                            if "type" in var_def:
                                var_type = var_def["type"]
                                if var_type not in ["string", "int", "float"]:
                                    errors.append(
                                        ValidationError(
                                            f"Invalid variable type '{var_type}'. Must be 'string', 'int', or 'float'",
                                            f"/user/goals/{i}/{var_name}/type",
                                        )
                                    )

                            # Validate function
                            if "function" in var_def:
                                func = var_def["function"]

                                # Basic function validation
                                valid_function = False

                                # Check for default(), another(), forward() without parameters
                                if func in ["default()", "another()", "forward()"]:
                                    valid_function = True

                                # Check for random() with different formats
                                elif func == "random()":
                                    valid_function = True
                                elif func.startswith("random(") and func.endswith(")"):
                                    # Extract the parameter
                                    random_length = len("random(")
                                    param = func[random_length:-1]
                                    if param == "rand" or param.isdigit():
                                        valid_function = True

                                # CHeck for another() with parameters
                                elif func.startswith("another(") and func.endswith(")"):
                                    # Extract the parameter
                                    another_length = len("another(")
                                    param = func[another_length:-1]
                                    if param.isdigit():
                                        valid_function = True

                                # Check for forward with nested variable
                                elif func.startswith("forward(") and func.endswith(")"):
                                    nested_length = len("forward(")
                                    nested_var = func[nested_length:-1]
                                    if (
                                        nested_var in defined_variables
                                        or nested_var == ""
                                    ):
                                        valid_function = True
                                    else:
                                        errors.append(
                                            ValidationError(
                                                f"Forward function references undefined variable '{nested_var}'",
                                                f"/user/goals/{i}/{var_name}/function",
                                            )
                                        )
                                        valid_function = True  # Avoid duplicate error

                                if not valid_function:
                                    errors.append(
                                        ValidationError(
                                            f"Invalid function '{func}'. Must be one of: default(), random(), random(n), random(rand), another(), forward() or forward(var)",
                                            f"/user/goals/{i}/{var_name}/function",
                                        )
                                    )

                            # Validate data structure based on type
                            if "data" in var_def and "type" in var_def:
                                data = var_def["data"]
                                var_type = var_def["type"]

                                # For numeric types with range definition
                                if var_type in ["int", "float"] and isinstance(
                                    data, dict
                                ):
                                    # Check for min/max
                                    if "min" not in data:
                                        errors.append(
                                            ValidationError(
                                                "Missing 'min' in numeric range definition",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )
                                    if "max" not in data:
                                        errors.append(
                                            ValidationError(
                                                "Missing 'max' in numeric range definition",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )
                                    # Check that min is smaller than max
                                    if (
                                        "min" in data
                                        and "max" in data
                                        and data["min"] is not None
                                        and data["max"] is not None
                                    ):
                                        if data["min"] >= data["max"]:
                                            errors.append(
                                                ValidationError(
                                                    "Minimum value must be smaller than maximum value",
                                                    f"/user/goals/{i}/{var_name}/data",
                                                )
                                            )

                                    # Check that either step or linspace is provided for float
                                    if (
                                        var_type == "float"
                                        and "step" not in data
                                        and "linspace" not in data
                                    ):
                                        errors.append(
                                            ValidationError(
                                                "Float range must define either 'step' or 'linspace'",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )

                                    # For int, check that step is provided
                                    if var_type == "int" and "step" not in data:
                                        errors.append(
                                            ValidationError(
                                                "Integer range must define 'step'",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )

                                # For list data, check that it's actually a list
                                elif not isinstance(data, list) and not isinstance(
                                    data, dict
                                ):
                                    errors.append(
                                        ValidationError(
                                            "Data must be a list, range definition, or custom function",
                                            f"/user/goals/{i}/{var_name}/data",
                                        )
                                    )

                                # For custom function data
                                elif isinstance(data, dict) and "file" in data:
                                    if "function_name" not in data:
                                        errors.append(
                                            ValidationError(
                                                "Custom function data must include 'function_name'",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )
                                    if "args" not in data:
                                        errors.append(
                                            ValidationError(
                                                "Custom function data must include 'args'",
                                                f"/user/goals/{i}/{var_name}/data",
                                            )
                                        )

                                # For list data, validate any() functions if present
                                elif isinstance(data, list):
                                    for j, item in enumerate(data):
                                        if isinstance(item, str):
                                            # Check for any() function format
                                            if item.startswith("any("):
                                                if not item.endswith(")"):
                                                    errors.append(
                                                        ValidationError(
                                                            f"Malformed any() function: Missing closing parenthesis in '{item}'",
                                                            f"/user/goals/{i}/{var_name}/data/{j}",
                                                        )
                                                    )
                                                elif (
                                                    len(item) <= 5
                                                ):  # "any()" has length 5
                                                    errors.append(
                                                        ValidationError(
                                                            "Empty any() function: Must contain instructions",
                                                            f"/user/goals/{i}/{var_name}/data/{j}",
                                                        )
                                                    )
                                                # Check for balanced parentheses within any()
                                                elif item.count("(") != item.count(")"):
                                                    errors.append(
                                                        ValidationError(
                                                            f"Unbalanced parentheses in any() function: '{item}'",
                                                            f"/user/goals/{i}/{var_name}/data/{j}",
                                                        )
                                                    )
                                        elif not isinstance(
                                            item, (str, int, float, bool)
                                        ):
                                            errors.append(
                                                ValidationError(
                                                    f"Invalid data list item type: {type(item).__name__}. Must be a primitive value or any() function",
                                                    f"/user/goals/{i}/{var_name}/data/{j}",
                                                )
                                            )

                    else:
                        errors.append(
                            ValidationError(
                                "Goals must be either strings or dictionaries",
                                f"/user/goals/{i}",
                            )
                        )

            # Third pass: validate forward dependencies
            forward_dependencies = {}
            for var_name, var_info in defined_variables.items():
                # Skip if the variable doesn't have a function defined
                if not var_info["function"]:
                    continue

                func = var_info["function"]
                # Check if this is a forward function with a variable reference
                if (
                    func.startswith("forward(")
                    and func.endswith(")")
                    and func != "forward()"
                ):
                    # Extract the referenced variable name
                    referenced_var = func[len("forward(") : -1].strip()
                    if referenced_var:
                        forward_dependencies[var_name] = referenced_var

            # Validate each forward dependency chain
            for var_name, referenced_var in forward_dependencies.items():
                # Skip if the referenced variable doesn't exist (already caught in other validations)
                if referenced_var not in defined_variables:
                    continue

                # Check if the referenced variable also uses forward
                ref_function = defined_variables[referenced_var]["function"]
                if ref_function and not ref_function.startswith("forward("):
                    errors.append(
                        ValidationError(
                            f"Variable '{referenced_var}' is referenced by forward() but doesn't use forward() itself",
                            f"/user/goals/{defined_variables[referenced_var]['index']}/{referenced_var}/function",
                        )
                    )

            # Detect circular dependencies
            def detect_cycle(node, visited, path, cycles):
                visited[node] = True
                path.append(node)

                if node in forward_dependencies:
                    next_node = forward_dependencies[node]
                    if next_node in path:
                        # Found a cycle, extract the cycle path
                        cycle_start = path.index(next_node)
                        cycle_path = path[cycle_start:] + [next_node]
                        cycles.append(" → ".join(cycle_path))
                    elif next_node not in visited or not visited[next_node]:
                        detect_cycle(next_node, visited, path, cycles)

                path.pop()

            # Find all cycles in the dependencies
            visited = {var: False for var in defined_variables}
            cycles = []

            for var in forward_dependencies:
                if not visited[var]:
                    detect_cycle(var, visited, [], cycles)

            # Report any cycles found
            if cycles:
                cycle_descriptions = "; ".join(cycles)
                errors.append(
                    ValidationError(
                        f"Circular forward dependencies detected: {cycle_descriptions}. Forward references must form a directed acyclic graph.",
                        "/user/goals",
                    )
                )

        return errors

    def _validate_chatbot_section(self, chatbot: Dict) -> List[ValidationError]:
        """Validate chatbot section configuration."""
        errors = []

        # is_start can only be true or false
        if "is_starter" in chatbot:
            is_starter = chatbot["is_starter"]
            if not isinstance(is_starter, bool):
                errors.append(
                    ValidationError(
                        "is_starter must be a boolean (true or false)",
                        "/chatbot/is_starter",
                    )
                )

        # fallback can only be string
        if "fallback" in chatbot:
            fallback = chatbot["fallback"]
            if not isinstance(fallback, str):
                errors.append(
                    ValidationError("Fallback must be a string", "/chatbot/fallback")
                )

        if "output" in chatbot:
            if not isinstance(chatbot["output"], list):
                errors.append(
                    ValidationError("Output must be a list", "/chatbot/output")
                )
            else:
                for i, output_item in enumerate(chatbot["output"]):
                    # Each output item should be a dictionary with a single key (output name)
                    if not isinstance(output_item, dict):
                        errors.append(
                            ValidationError(
                                "Each output item must be a dictionary",
                                f"/chatbot/output/{i}",
                            )
                        )
                        continue

                    # Check if output item has exactly one key (output name)
                    if len(output_item) != 1:
                        errors.append(
                            ValidationError(
                                "Each output item must have exactly one key (the output name)",
                                f"/chatbot/output/{i}",
                            )
                        )
                        continue

                    output_name = list(output_item.keys())[0]
                    output_def = output_item[output_name]

                    # Check that output definition is a dictionary
                    if not isinstance(output_def, dict):
                        errors.append(
                            ValidationError(
                                f"Output definition for '{output_name}' must be a dictionary",
                                f"/chatbot/output/{i}/{output_name}",
                            )
                        )
                        continue

                    # Check required fields: type and description
                    if "type" not in output_def:
                        errors.append(
                            ValidationError(
                                f"Output '{output_name}' must have a type",
                                f"/chatbot/output/{i}/{output_name}",
                            )
                        )

                    if "description" not in output_def:
                        errors.append(
                            ValidationError(
                                f"Output '{output_name}' must have a description",
                                f"/chatbot/output/{i}/{output_name}",
                            )
                        )

                    # Validate output type
                    if (
                        "type" in output_def
                        and output_def["type"] not in self.valid_output_types
                    ):
                        errors.append(
                            ValidationError(
                                f"Invalid output type '{output_def['type']}'. Must be one of: {', '.join(self.valid_output_types)}",
                                f"/chatbot/output/{i}/{output_name}/type",
                            )
                        )

                    # Validate description is a non-empty string
                    if "description" in output_def:
                        desc = output_def["description"]
                        if not isinstance(desc, str) or desc.strip() == "":
                            errors.append(
                                ValidationError(
                                    "Description must be a non-empty string",
                                    f"/chatbot/output/{i}/{output_name}/description",
                                )
                            )

        return errors

    def _validate_conversation_section(
        self, conversation: Dict
    ) -> List[ValidationError]:
        """Validate conversation section configuration."""
        errors = []

        # Validate 'number' field - can be integer, 'all_combinations', or sample()
        if "number" in conversation:
            num = conversation["number"]
            if isinstance(num, int):
                if num <= 0:
                    errors.append(
                        ValidationError(
                            "Number of conversations must be a positive integer",
                            "/conversation/number",
                        )
                    )
            elif isinstance(num, str):
                # Check for "all_combinations" string
                if num == "all_combinations":
                    # This is valid
                    pass
                # Check for sample() function format
                elif num.startswith("sample(") and num.endswith(")"):
                    # Extract the parameter
                    sample_param = num[len("sample(") : -1]
                    try:
                        sample_value = float(sample_param)
                        if sample_value <= 0 or sample_value > 1:
                            errors.append(
                                ValidationError(
                                    "Sample value must be between 0 and 1",
                                    "/conversation/number",
                                )
                            )
                    except ValueError:
                        errors.append(
                            ValidationError(
                                "Invalid sample value, must be a decimal between 0 and 1",
                                "/conversation/number",
                            )
                        )
                else:
                    errors.append(
                        ValidationError(
                            "Number must be a positive integer, 'all_combinations', or sample(0.0-1.0)",
                            "/conversation/number",
                        )
                    )
            else:
                errors.append(
                    ValidationError(
                        "Number must be a positive integer, 'all_combinations', or sample(0.0-1.0)",
                        "/conversation/number",
                    )
                )

        if "max_cost" in conversation:
            cost = conversation["max_cost"]
            if not isinstance(cost, (int, float)) or cost <= 0:
                errors.append(
                    ValidationError(
                        "Max cost must be a positive number", "/conversation/max_cost"
                    )
                )

        # Validate goal_style
        if "goal_style" in conversation:
            goal_style = conversation["goal_style"]

            # goal_style can be either a string ("default") or a dictionary
            if isinstance(goal_style, str):
                # Only "default" is a valid string value
                if goal_style != "default":
                    errors.append(
                        ValidationError(
                            "When goal_style is a string, it must be 'default'",
                            "/conversation/goal_style",
                        )
                    )
            elif isinstance(goal_style, dict):
                # Check for valid goal style options
                for key in goal_style:
                    if key not in self.valid_goal_styles:
                        errors.append(
                            ValidationError(
                                f"Invalid goal_style option: {key}\nValid options: {', '.join(self.valid_goal_styles)}",
                                f"/conversation/goal_style/{key}",
                            )
                        )

                # Validate steps (must be positive integer)
                if "steps" in goal_style:
                    steps = goal_style["steps"]
                    if not isinstance(steps, int) or steps <= 0:
                        errors.append(
                            ValidationError(
                                "Steps must be a positive integer",
                                "/conversation/goal_style/steps",
                            )
                        )

                # Validate random_steps (must be positive integer <= 20)
                if "random_steps" in goal_style:
                    random_steps = goal_style["random_steps"]
                    if not isinstance(random_steps, int) or random_steps <= 0:
                        errors.append(
                            ValidationError(
                                "Random steps must be a positive integer",
                                "/conversation/goal_style/random_steps",
                            )
                        )
                    elif random_steps > 20:
                        errors.append(
                            ValidationError(
                                "Random steps cannot exceed 20",
                                "/conversation/goal_style/random_steps",
                            )
                        )

                # Validate all_answered
                if "all_answered" in goal_style:
                    all_answered = goal_style["all_answered"]

                    # all_answered can be boolean or dictionary
                    if isinstance(all_answered, bool):
                        # Boolean value is fine
                        pass
                    elif isinstance(all_answered, dict):
                        # Validate export field if present, is optional
                        if "export" in all_answered and not isinstance(
                            all_answered["export"], bool
                        ):
                            errors.append(
                                ValidationError(
                                    "Export field must be a boolean",
                                    "/conversation/goal_style/all_answered/export",
                                )
                            )

                        # Validate limit field if present
                        if "limit" in all_answered:
                            limit = all_answered["limit"]
                            if not isinstance(limit, int) or limit <= 0:
                                errors.append(
                                    ValidationError(
                                        "Limit must be a positive integer",
                                        "/conversation/goal_style/all_answered/limit",
                                    )
                                )
                    else:
                        errors.append(
                            ValidationError(
                                "all_answered must be a boolean or a dictionary",
                                "/conversation/goal_style/all_answered",
                            )
                        )

                # Validate max_cost (per conversation cost limit), optional
                if "max_cost" in goal_style:
                    cost = goal_style["max_cost"]
                    if not isinstance(cost, (int, float)) or cost <= 0:
                        errors.append(
                            ValidationError(
                                "Goal style max_cost must be a positive number",
                                "/conversation/goal_style/max_cost",
                            )
                        )
            else:
                errors.append(
                    ValidationError(
                        "Goal style must be either 'default' or a dictionary",
                        "/conversation/goal_style",
                    )
                )

        # Validate interaction_style
        if "interaction_style" in conversation:
            interaction_style = conversation["interaction_style"]

            if not isinstance(interaction_style, list):
                errors.append(
                    ValidationError(
                        "Interaction style must be a list",
                        "/conversation/interaction_style",
                    )
                )
            else:
                for i, style in enumerate(interaction_style):
                    # Simple string style
                    if isinstance(style, str):
                        if style == "change language":
                            errors.append(
                                ValidationError(
                                    "'change language' must be a dictionary with a list of languages",
                                    f"/conversation/interaction_style/{i}",
                                )
                            )
                        elif style not in self.valid_interaction_styles:
                            errors.append(
                                ValidationError(
                                    f"Invalid interaction style: '{style}'. Must be one of: {', '.join(self.valid_interaction_styles)}",
                                    f"/conversation/interaction_style/{i}",
                                )
                            )

                    # Complex dictionary style
                    elif isinstance(style, dict):
                        if len(style) != 1:
                            errors.append(
                                ValidationError(
                                    "Each complex style must have exactly one key",
                                    f"/conversation/interaction_style/{i}",
                                )
                            )
                            continue

                        style_type = list(style.keys())[0]
                        style_value = style[style_type]

                        # Handle "random" style - must be a list of other styles
                        if style_type == "random":
                            if not isinstance(style_value, list):
                                errors.append(
                                    ValidationError(
                                        "Random style must be a list of other styles",
                                        f"/conversation/interaction_style/{i}/random",
                                    )
                                )
                                continue

                            # Check each random style option
                            for j, random_style in enumerate(style_value):
                                if isinstance(random_style, str):
                                    if random_style == "change language":
                                        errors.append(
                                            ValidationError(
                                                "'change language' must be a dictionary with a list of languages",
                                                f"/conversation/interaction_style/{i}/random/{j}",
                                            )
                                        )
                                    elif (
                                        random_style
                                        not in self.valid_interaction_styles
                                    ):
                                        errors.append(
                                            ValidationError(
                                                f"Invalid style in random list: '{random_style}'",
                                                f"/conversation/interaction_style/{i}/random/{j}",
                                            )
                                        )
                                elif isinstance(random_style, dict):
                                    # Handle nested styles within random list
                                    if (
                                        len(random_style) != 1
                                        or "change language" not in random_style
                                    ):
                                        errors.append(
                                            ValidationError(
                                                "Only 'change language' can be a nested dictionary in random list",
                                                f"/conversation/interaction_style/{i}/random/{j}",
                                            )
                                        )
                                    else:
                                        lang_list = random_style["change language"]
                                        if not isinstance(lang_list, list):
                                            errors.append(
                                                ValidationError(
                                                    "Change language must specify a list of languages",
                                                    f"/conversation/interaction_style/{i}/random/{j}/change language",
                                                )
                                            )
                                        else:
                                            # Just check that each item is a string
                                            for k, lang in enumerate(lang_list):
                                                if not isinstance(lang, str):
                                                    errors.append(
                                                        ValidationError(
                                                            f"Language must be a string, got {type(lang).__name__}",
                                                            f"/conversation/interaction_style/{i}/random/{j}/change language/{k}",
                                                        )
                                                    )
                                else:
                                    errors.append(
                                        ValidationError(
                                            "Random style items must be strings or 'change language' dictionary",
                                            f"/conversation/interaction_style/{i}/random/{j}",
                                        )
                                    )

                        # Handle "change language" style - must be a list of languages
                        elif style_type == "change language":
                            if not isinstance(style_value, list):
                                errors.append(
                                    ValidationError(
                                        "Change language must be a list of languages",
                                        f"/conversation/interaction_style/{i}/change language",
                                    )
                                )
                                continue

                            # Check each language is a string (no validation of specific languages)
                            for j, language in enumerate(style_value):
                                if not isinstance(language, str):
                                    errors.append(
                                        ValidationError(
                                            f"Language must be a string, got {type(language).__name__}",
                                            f"/conversation/interaction_style/{i}/change language/{j}",
                                        )
                                    )

                        # Any other complex style is invalid
                        else:
                            errors.append(
                                ValidationError(
                                    f"Invalid complex style: '{style_type}'. Must be 'random' or 'change language'",
                                    f"/conversation/interaction_style/{i}",
                                )
                            )

                    # Not a string or dictionary
                    else:
                        errors.append(
                            ValidationError(
                                f"Interaction style item must be a string or dictionary, got {type(style).__name__}",
                                f"/conversation/interaction_style/{i}",
                            )
                        )
        return errors


# Usage example:
if __name__ == "__main__":
    validator = YamlValidator()

    # Example YAML content
    yaml_content = """
test_name: "pizza_order_test_all"

llm:
  temperature: 0.8
  model: gpt-4o-mini
#  format:
#    type: speech
#    config: asr_configuration/default_asr_config.yml

user:
  language: English
  role: you have to act as a user ordering a pizza to a pizza shop.
  context:
    - personality: personalities/conversational-user.yml
    - your name is Jon Doe
  goals:
    - "a {{size}} custom pizza with toppings, {{size}}"
    - how long is going to take the pizza to arrive
    - how much will it cost

    - size:
        function: forward(toppings)
        type: string
        data:
          - small
          - medium
          - big

    - toppings:
        function: forward()
        type: string
        data:
          - cheese
          - mushrooms
          - pepperoni

chatbot:
  is_starter: False
  fallback: I'm sorry it's a little loud in my pizza shop, can you say that again?
  output:
    - price:
        type: money
        description: The final price of the pizza order
    - time:
        type: time
        description: how long is going to take the pizza to be ready
    - order_id:
        type: str
        description: my order ID

conversation:
  number: all_combinations
  goal_style:
    steps: 2
  interaction_style:
    - random:
      - make spelling mistakes
    """
    print("Starting...")

    errors = validator.validate(yaml_content)
    for error in errors:
        print(f"Error at {error.path}: {error.message}")

    if not errors:
        print("No errors found.")
