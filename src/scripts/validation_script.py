"""Script to validate the profiles for the user simulator."""

import re
from dataclasses import dataclass

import yaml


@dataclass
class ValidationError:
    """Represents a single validation error found in the YAML content.

    Args:
        message: The description of the error.
        path: The JSON path-like location of the error in the YAML structure.
        line: The line number where the error occurred (optional).
    """

    message: str
    path: str
    line: int | None = None


languages = [
    "Afrikaans",
    "Albanian",
    "Amharic",
    "Arabic",
    "Armenian",
    "Azerbaijani",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Catalan",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Filipino",
    "Finnish",
    "French",
    "Galician",
    "Georgian",
    "German",
    "Greek",
    "Gujarati",
    "Hausa",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kannada",
    "Kazakh",
    "Korean",
    "Latvian",
    "Lithuanian",
    "Macedonian",
    "Malay",
    "Malayalam",
    "Marathi",
    "Nepali",
    "Norwegian",
    "Persian",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "Serbian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tamil",
    "Telugu",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Vietnamese",
    "Zulu",
]


class YamlValidator:
    """Validates YAML configuration files against a predefined schema.

    Attributes:
        required_top_level: List of mandatory top-level keys.
        required_nested: Dictionary mapping nested paths to lists of mandatory keys within them.
        allowed_fields: Dictionary mapping sections to lists of allowed keys within them.
        valid_formats: List of valid values for the 'llm.format.type' field.
        valid_goal_functions: List of valid function signatures for user goals.
        valid_variable_types: List of valid types for user goal variables.
        valid_output_types: List of valid types for chatbot outputs.
        valid_goal_styles: List of valid keys for the 'conversation.goal_style' dictionary.
        valid_interaction_styles: List of valid string values for 'conversation.interaction_style'.
    """

    def __init__(self) -> None:
        """Initializes the YamlValidator with predefined schema rules."""
        self.required_top_level = [
            "test_name",
            # "llm",
            "user",
            "conversation",
            "chatbot",
        ]
        # These are mandatory, if they are not present, the validation will fail
        self.required_nested = {
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

    def validate(self, yaml_content: str) -> list[ValidationError]:
        """Validate YAML content against schema rules.

        Checks that all required fields are present, have the correct type,
        and adhere to specific constraints defined in the validator.

        Args:
            yaml_content: The YAML content as a string.

        Returns:
            A list of ValidationError objects, empty if validation passes.
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
                errors.extend(self._validate_conversation_variable_dependencies(data["user"], data["conversation"]))

        except yaml.YAMLError as e:
            return [ValidationError(f"Invalid YAML syntax: {e!s} at line {e.problem_mark.line + 1}", "/")]
        else:
            return errors

    def _validate_conversation_variable_dependencies(self, user: dict, conversation: dict) -> list[ValidationError]:
        """Validate dependencies between conversation settings and user goal variables.

        Ensures 'all_combinations' or 'sample()' in conversation.number is used
        only when there's at least one user goal variable with a nested forward() dependency.

        Args:
            user: The 'user' section dictionary from the YAML data.
            conversation: The 'conversation' section dictionary from the YAML data.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # Check if conversation.number is using sample() or all_combinations
        using_combinations = False
        if "number" in conversation:
            num = conversation["number"]
            if isinstance(num, str) and (num == "all_combinations" or num.startswith("sample(")):
                using_combinations = True

        # If using combinations, check for nested forwards
        if using_combinations:
            has_nested_forwards = False

            # Find all forwards in the user goals
            if "goals" in user and isinstance(user["goals"], list):
                for goal in user["goals"]:
                    if isinstance(goal, dict) and len(goal) == 1:
                        var_name = next(iter(goal.keys()))
                        var_def = goal[var_name]

                        if isinstance(var_def, dict) and "function" in var_def:
                            func = var_def["function"]
                            # Check if this is a forward function referring to another variable
                            if func.startswith("forward(") and func.endswith(")") and func != "forward()":
                                has_nested_forwards = True
                                break

            # If using combinations but no nested forwards, show an error
            if not has_nested_forwards:
                errors.append(
                    ValidationError(
                        "Using 'all_combinations' or 'sample()' requires at least one variable with nested forward() dependency",
                        "/conversation/number",
                    ),
                )

        return errors

    def _validate_required_fields(self, data: dict) -> list[ValidationError]:
        """Validate that all required top-level and nested fields are present.

        Args:
            data: The dictionary representing the loaded YAML data.

        Returns:
            A list of ValidationError objects for missing required fields.
        """
        errors = [
            ValidationError(f"Missing required field: {field}", f"/{field}")
            for field in self.required_top_level
            if field not in data
        ]

        # Check nested required fields
        for path, fields in self.required_nested.items():
            parts = path.split(".")
            current = data
            current_path = ""
            valid_path = True
            for part in parts:
                current_path += f"/{part}"
                if not isinstance(current, dict) or part not in current:
                    # Don't report missing nested fields if the parent doesn't exist or isn't a dict
                    valid_path = False
                    break
                current = current[part]

            if valid_path:
                for field in fields:
                    if not isinstance(current, dict) or field not in current:
                        errors.append(
                            ValidationError(
                                f"Missing required field: {field} in {path}",
                                f"{current_path}/{field}",
                            ),
                        )

        return errors

    def _validate_llm_section(self, llm: dict) -> list[ValidationError]:
        """Validate the 'llm' section configuration.

        Args:
            llm: The 'llm' section dictionary from the YAML data.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # Check for unexpected fields in the llm section
        for field in llm:
            if field not in self.allowed_fields["llm"]:
                errors.append(
                    ValidationError(
                        f"Unexpected field '{field}' in llm section. Allowed fields: {', '.join(self.allowed_fields['llm'])}",
                        f"/llm/{field}",
                    ),
                )

        # Check that temperature is a number between 0 and 1
        if "temperature" in llm:
            temp = llm["temperature"]
            if not isinstance(temp, (int, float)) or not (0 <= temp <= 1):
                errors.append(
                    ValidationError(
                        "Temperature must be a number between 0 and 1 (inclusive)",
                        "/llm/temperature",
                    ),
                )

        # Validate format section
        if "format" in llm:
            if not isinstance(llm["format"], dict):
                errors.append(ValidationError("llm.format must be a dictionary", "/llm/format"))
            else:
                format_section = llm["format"]
                format_type = format_section.get("type")

                # Validate format type (required field checked elsewhere)
                if format_type and format_type not in self.valid_formats:
                    errors.append(
                        ValidationError(
                            f"Invalid format type '{format_type}'. Must be one of: {', '.join(self.valid_formats)}",
                            "/llm/format/type",
                        ),
                    )

                # If type is speech, check for config
                if format_type == "speech":
                    config_val = format_section.get("config")
                    if config_val is None or not isinstance(config_val, str) or config_val.strip() == "":
                        errors.append(
                            ValidationError(
                                "Speech format requires a non-empty 'config' field with path to configuration file",
                                "/llm/format/config",
                            ),
                        )

        return errors

    def _validate_user_section(self, user: dict) -> list[ValidationError]:
        """Validate the 'user' section configuration.

        Args:
            user: The 'user' section dictionary from the YAML data.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # Validate context structure (required field checked elsewhere)
        if "context" in user:
            errors.extend(self._validate_user_context(user["context"]))

        # Validate language (optional field)
        if "language" in user and user["language"] not in languages:
            errors.append(
                ValidationError(
                    message=(f"Invalid language '{user['language']}'. Must be one of: {', '.join(languages)}"),
                    path="/user/language",
                ),
            )

        # Validate goals structure (required field checked elsewhere)
        if "goals" in user:
            errors.extend(self._validate_user_goals(user["goals"]))

        return errors

    def _validate_user_context(self, context: list) -> list[ValidationError]:
        """Validate the 'user.context' list.

        Args:
            context: The list value of the 'user.context' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if not isinstance(context, list):
            errors.append(ValidationError("User context must be a list", "/user/context"))
            return errors  # Stop further context validation if it's not a list

        has_personality = False
        for i, item in enumerate(context):
            path = f"/user/context/{i}"
            if isinstance(item, dict):
                if "personality" in item:
                    if has_personality:
                        errors.append(ValidationError("Only one personality entry is allowed in context", path))
                    has_personality = True
                    # Validate that personality points to a non-empty string path
                    if not isinstance(item["personality"], str) or item["personality"].strip() == "":
                        errors.append(
                            ValidationError(
                                "Personality must be a non-empty string path",
                                f"{path}/personality",
                            ),
                        )
                    # Check for other keys in the personality dict
                    if len(item) > 1:
                        extra_keys = [k for k in item if k != "personality"]
                        errors.append(
                            ValidationError(
                                f"Personality entry should only contain the 'personality' key. Found extra keys: {', '.join(extra_keys)}",
                                path,
                            ),
                        )
                else:
                    errors.append(
                        ValidationError(
                            "Dictionary items in context must be a personality entry (e.g., {'personality': 'path/to/file.yml'})",
                            path,
                        ),
                    )
            elif not isinstance(item, str):
                errors.append(
                    ValidationError(
                        f"Context items must be strings or a personality dictionary, got {type(item).__name__}",
                        path,
                    ),
                )
            elif isinstance(item, str) and item.strip() == "":
                errors.append(ValidationError("Context string items cannot be empty", path))

        return errors

    def _validate_user_goals(self, goals: list) -> list[ValidationError]:
        """Validate the 'user.goals' list.

        Args:
            goals: The list value of the 'user.goals' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if not isinstance(goals, list):
            errors.append(ValidationError("User goals must be a list", "/user/goals"))
            return errors  # Stop further goals validation if it's not a list

        # First pass: collect all defined variables and their functions
        defined_variables = {}
        for i, goal in enumerate(goals):
            if isinstance(goal, dict) and len(goal) == 1:
                var_name = next(iter(goal.keys()))
                var_def = goal[var_name]
                # Basic check for variable name format (alphanumeric + underscore)
                if not re.fullmatch(r"[a-zA-Z0-9_]+", var_name):
                    errors.append(
                        ValidationError(
                            f"Invalid variable name '{var_name}'. Use only letters, numbers, and underscores.",
                            f"/user/goals/{i}",
                        ),
                    )

                func = None
                if isinstance(var_def, dict) and "function" in var_def:
                    func = var_def.get("function")

                defined_variables[var_name] = {"function": func, "index": i}
            elif (isinstance(goal, dict) and len(goal) != 1) or not isinstance(goal, str):
                # Error handled in the second pass
                pass

        # Second pass: validate individual goals (strings and variable definitions)
        for i, goal in enumerate(goals):
            path = f"/user/goals/{i}"
            if isinstance(goal, str):
                errors.extend(self._validate_goal_string(goal, defined_variables, path))
            elif isinstance(goal, dict):
                errors.extend(self._validate_goal_variable_definition(goal, defined_variables, path, i))
            else:
                errors.append(
                    ValidationError(
                        f"Goals must be either strings (prompts) or dictionaries (variable definitions), got {type(goal).__name__}",
                        path,
                    ),
                )

        # Third pass: validate forward dependencies and detect cycles
        errors.extend(self._validate_forward_dependencies(defined_variables))

        return errors

    def _validate_goal_string(self, goal_str: str, defined_variables: dict, path: str) -> list[ValidationError]:
        """Validate a string goal (prompt) for correct variable usage.

        Args:
            goal_str: The string goal content.
            defined_variables: Dictionary of variables defined in the goals section.
            path: The JSON path to this goal string.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        # Regex to find valid placeholders like {{var_name}} or {{ var name }}
        # It captures the content inside the braces.
        placeholder_pattern = r"{{\s*([^{}]+?)\s*}}"
        valid_vars_found = re.findall(placeholder_pattern, goal_str)

        # Check for invalid curly brace usage by removing valid placeholders
        # and then looking for remaining braces.
        temp_str = re.sub(placeholder_pattern, "", goal_str)
        if "{" in temp_str or "}" in temp_str:
            # Try to find the first occurrence of an invalid pattern
            invalid_match = re.search(r"([^{]*\{[^{}]*?\})|([^{}]*?\}[^}]*)", temp_str)
            context = invalid_match.group(0).strip() if invalid_match else "around braces"
            errors.append(
                ValidationError(
                    f"Invalid use of curly braces near '{context}'. Use exactly two opening and two closing braces for variables: '{{ variable_name }}'.",
                    path,
                ),
            )

        # Check if the variables used inside valid placeholders are defined
        # and match the expected format (alphanumeric + underscore).
        for var_content in valid_vars_found:
            var_name = var_content.strip()  # Get the actual variable name
            if not re.fullmatch(r"[a-zA-Z0-9_]+", var_name):
                errors.append(
                    ValidationError(
                        f"Invalid variable name format '{var_name}' used in goal. Use only letters, numbers, and underscores.",
                        path,
                    ),
                )
            elif var_name not in defined_variables:
                errors.append(
                    ValidationError(
                        f"Variable '{{{{ {var_name} }}}}' used in goal but not defined in the goals list.",
                        path,
                    ),
                )
        return errors

    def _validate_goal_variable_definition(
        self, goal_dict: dict, defined_variables: dict, path: str, index: int
    ) -> list[ValidationError]:
        """Validate a dictionary goal (variable definition).

        Args:
            goal_dict: The dictionary representing the variable definition.
            defined_variables: Dictionary of variables defined in the goals section.
            path: The JSON path to this goal dictionary.
            index: The index of this goal in the goals list.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if len(goal_dict) != 1:
            errors.append(
                ValidationError("Each variable definition must have exactly one key (the variable name)", path)
            )
            return errors  # Cannot proceed if structure is wrong

        var_name = next(iter(goal_dict.keys()))
        var_def = goal_dict[var_name]
        var_path = f"{path}/{var_name}"

        if not isinstance(var_def, dict):
            errors.append(ValidationError(f"Variable '{var_name}' definition must be a dictionary", var_path))
            return errors  # Cannot proceed if definition is not a dict

        # Check required fields within the variable definition
        for field in ["function", "type", "data"]:
            if field not in var_def:
                errors.append(ValidationError(f"Missing required field '{field}' in variable '{var_name}'", var_path))

        # Validate 'type' field
        if "type" in var_def:
            var_type = var_def["type"]
            if var_type not in self.valid_variable_types:
                errors.append(
                    ValidationError(
                        f"Invalid variable type '{var_type}'. Must be one of: {', '.join(self.valid_variable_types)}",
                        f"{var_path}/type",
                    ),
                )

        # Validate 'function' field
        if "function" in var_def:
            func = var_def["function"]
            errors.extend(self._validate_variable_function(func, defined_variables, var_name, f"{var_path}/function"))

        # Validate 'data' structure based on 'type'
        if "data" in var_def and "type" in var_def:
            data = var_def["data"]
            var_type = var_def.get("type")  # Use get to avoid error if type is missing
            errors.extend(self._validate_variable_data(data, var_type, f"{var_path}/data"))

        return errors

    def _validate_variable_function(
        self, func: str, defined_variables: dict, current_var: str, path: str
    ) -> list[ValidationError]:
        """Validate the 'function' field within a variable definition.

        Args:
            func: The function string (e.g., "default()", "forward(other_var)").
            defined_variables: Dictionary of all defined variables.
            current_var: The name of the variable this function belongs to.
            path: The JSON path to the function field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if not isinstance(func, str):
            errors.append(ValidationError(f"Function must be a string, got {type(func).__name__}", path))
            return errors  # Cannot validate non-string function

        valid_function = False
        # Simple functions without parameters or with specific allowed parameters
        if func in ["default()", "random()", "another()", "forward()"]:
            valid_function = True

        elif func.startswith("random(") and func.endswith(")"):
            param = func[len("random(") : -1].strip()
            if param == "rand" or param.isdigit():
                valid_function = True

        elif func.startswith("another(") and func.endswith(")"):
            param = func[len("another(") : -1].strip()
            if param.isdigit():
                valid_function = True

        elif func.startswith("forward(") and func.endswith(")"):
            nested_var = func[len("forward(") : -1].strip()
            if nested_var == "":  # forward() case already handled
                pass
            elif nested_var == current_var:
                errors.append(ValidationError(f"Forward function cannot reference itself ('{nested_var}')", path))
                valid_function = True  # Mark as valid to avoid generic invalid function error
            elif nested_var not in defined_variables:
                errors.append(ValidationError(f"Forward function references undefined variable '{nested_var}'", path))
                valid_function = True  # Mark as valid to avoid generic invalid function error
            else:
                valid_function = True  # Valid forward reference

        if not valid_function and not errors:  # Only add generic error if no specific error was found
            allowed_funcs = (
                "default(), random(), random(n), random(rand), another(), another(n), forward(), forward(var_name)"
            )
            errors.append(ValidationError(f"Invalid function format '{func}'. Allowed formats: {allowed_funcs}", path))

        return errors

    def _validate_variable_data(self, data: any, var_type: str | None, path: str) -> list[ValidationError]:
        """Validate the 'data' field within a variable definition based on its 'type'.

        Args:
            data: The value of the 'data' field.
            var_type: The value of the 'type' field (string, int, float).
            path: The JSON path to the data field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # If type is missing or invalid, we cannot reliably validate data structure
        if var_type not in self.valid_variable_types:
            return errors

        # --- Validation for numeric types (int, float) ---
        if var_type in ["int", "float"]:
            if isinstance(data, dict):  # Range definition
                # Check for required min/max
                if "min" not in data:
                    errors.append(ValidationError("Missing 'min' in numeric range definition", path))
                if "max" not in data:
                    errors.append(ValidationError("Missing 'max' in numeric range definition", path))

                min_val = data.get("min")
                max_val = data.get("max")

                # Validate min/max types and relationship
                if min_val is not None and max_val is not None:
                    valid_types = (int, float)
                    if not isinstance(min_val, valid_types):
                        errors.append(
                            ValidationError(
                                f"Range 'min' must be a number, got {type(min_val).__name__}", f"{path}/min"
                            )
                        )
                    if not isinstance(max_val, valid_types):
                        errors.append(
                            ValidationError(
                                f"Range 'max' must be a number, got {type(max_val).__name__}", f"{path}/max"
                            )
                        )
                    elif isinstance(min_val, valid_types) and isinstance(max_val, valid_types) and min_val >= max_val:
                        errors.append(
                            ValidationError("Range 'min' value must be strictly smaller than 'max' value", path)
                        )

                # Validate step/linspace based on type
                if var_type == "float":
                    if "step" not in data and "linspace" not in data:
                        errors.append(ValidationError("Float range must define either 'step' or 'linspace'", path))
                    elif "step" in data and not isinstance(data["step"], (int, float)):
                        errors.append(
                            ValidationError(
                                f"Range 'step' must be a number, got {type(data['step']).__name__}", f"{path}/step"
                            )
                        )
                    elif "linspace" in data and (not isinstance(data["linspace"], int) or data["linspace"] <= 0):
                        errors.append(
                            ValidationError(
                                f"Range 'linspace' must be a positive integer, got {data['linspace']}",
                                f"{path}/linspace",
                            )
                        )

                elif var_type == "int":
                    if "step" not in data:
                        errors.append(ValidationError("Integer range must define 'step'", path))
                    elif "step" in data and (not isinstance(data["step"], int) or data["step"] <= 0):
                        errors.append(
                            ValidationError(
                                f"Integer range 'step' must be a positive integer, got {data['step']}", f"{path}/step"
                            )
                        )
                    if "linspace" in data:
                        errors.append(
                            ValidationError(
                                "Integer range cannot use 'linspace', use 'step' instead", f"{path}/linspace"
                            )
                        )

                # Check for unexpected keys in range definition
                allowed_range_keys = {"min", "max", "step", "linspace"}
                for key in data:
                    if key not in allowed_range_keys:
                        errors.append(
                            ValidationError(f"Unexpected key '{key}' in numeric range definition", f"{path}/{key}")
                        )

            elif isinstance(data, list):  # List of numbers
                for j, item in enumerate(data):
                    item_path = f"{path}/{j}"
                    if not isinstance(item, (int, float)):
                        errors.append(
                            ValidationError(
                                f"Invalid data list item type: {type(item).__name__}. Expected {var_type}", item_path
                            )
                        )
                    elif var_type == "int" and not isinstance(item, int):
                        errors.append(
                            ValidationError(f"Invalid data list item type: expected int, got float ({item})", item_path)
                        )

            # Custom function definition (common for all types)
            elif isinstance(data, dict) and "file" in data:
                errors.extend(self._validate_custom_function_data(data, path))

            else:  # Invalid data structure for numeric type
                errors.append(
                    ValidationError(
                        f"Data for type '{var_type}' must be a list of values or a range definition (min, max, step/linspace)",
                        path,
                    )
                )

        # --- Validation for string type ---
        elif var_type == "string":
            if isinstance(data, list):  # List of strings or any() functions
                for j, item in enumerate(data):
                    item_path = f"{path}/{j}"
                    if isinstance(item, str):
                        # Check for any() function format
                        if item.startswith("any("):
                            min_any_function_length = 3  # Length of "any()" with something inside
                            if not item.endswith(")"):
                                errors.append(
                                    ValidationError(
                                        f"Malformed any() function: Missing closing parenthesis in '{item}'", item_path
                                    )
                                )
                            elif len(item) < min_any_function_length:  # "any()"
                                errors.append(
                                    ValidationError(
                                        "Empty any() function: Must contain instructions between parentheses", item_path
                                    )
                                )
                            # Basic check for balanced parentheses within any()
                            elif item.count("(") != item.count(")"):
                                errors.append(
                                    ValidationError(f"Unbalanced parentheses in any() function: '{item}'", item_path)
                                )
                        # Allow empty strings in the list ""
                    elif not isinstance(item, (str, int, float, bool)):  # Allow primitives to be stringified
                        errors.append(
                            ValidationError(
                                f"Invalid data list item type: {type(item).__name__}. Must be a primitive value (string, int, float, bool) or any() function string",
                                item_path,
                            ),
                        )
            # Custom function definition (common for all types)
            elif isinstance(data, dict) and "file" in data:
                errors.extend(self._validate_custom_function_data(data, path))
            else:
                errors.append(
                    ValidationError(
                        "Data for type 'string' must be a list of values or a custom function definition", path
                    )
                )

        return errors

    def _validate_custom_function_data(self, data: dict, path: str) -> list[ValidationError]:
        """Validate the structure of a custom function data definition.

        Args:
            data: The dictionary representing the custom function data.
            path: The JSON path to this data dictionary.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if "function_name" not in data:
            errors.append(ValidationError("Custom function data must include 'function_name'", path))
        elif not isinstance(data["function_name"], str) or data["function_name"].strip() == "":
            errors.append(
                ValidationError("Custom function 'function_name' must be a non-empty string", f"{path}/function_name")
            )

        if "args" not in data:
            errors.append(ValidationError("Custom function data must include 'args' (can be an empty list/dict)", path))
        elif not isinstance(data["args"], (list, dict)):
            errors.append(
                ValidationError(
                    f"Custom function 'args' must be a list or dictionary, got {type(data['args']).__name__}",
                    f"{path}/args",
                )
            )

        if not isinstance(data["file"], str) or data["file"].strip() == "":
            errors.append(ValidationError("Custom function 'file' must be a non-empty string path", f"{path}/file"))

        # Check for unexpected keys
        allowed_keys = {"file", "function_name", "args"}
        for key in data:
            if key not in allowed_keys:
                errors.append(ValidationError(f"Unexpected key '{key}' in custom function definition", f"{path}/{key}"))
        return errors

    def _validate_forward_dependencies(self, defined_variables: dict) -> list[ValidationError]:
        """Validate forward() dependencies between variables for correctness and cycles.

        Args:
            defined_variables: Dictionary of variables defined in the goals section,
                               including their function and index.

        Returns:
            A list of ValidationError objects related to forward dependencies.
        """
        errors = []
        forward_dependencies = {}  # Maps var_name -> referenced_var_name

        # Build the dependency map and perform initial checks
        for var_name, var_info in defined_variables.items():
            func = var_info.get("function")
            if isinstance(func, str) and func.startswith("forward(") and func.endswith(")") and func != "forward()":
                referenced_var = func[len("forward(") : -1].strip()
                if referenced_var:  # Ignore empty forward() or invalid references already caught
                    forward_dependencies[var_name] = referenced_var

                    # Check if the referenced variable exists (redundant check, but safe)
                    if referenced_var not in defined_variables:
                        # This error is already caught by _validate_variable_function
                        continue

                    # Check if the referenced variable *also* uses forward()
                    ref_info = defined_variables[referenced_var]
                    ref_function = ref_info.get("function")
                    ref_path = f"/user/goals/{ref_info['index']}/{referenced_var}/function"

                    if not isinstance(ref_function, str) or not ref_function.startswith("forward("):
                        errors.append(
                            ValidationError(
                                f"Variable '{referenced_var}' is referenced by forward() in '{var_name}', but '{referenced_var}' itself does not use a forward() function.",
                                ref_path,
                            ),
                        )

        # Detect circular dependencies using Depth First Search
        path = set()
        visited = set()
        cycles_found = []  # Store descriptions of cycles found

        def detect_cycle_recursive(node: str) -> bool:
            path.add(node)
            visited.add(node)
            if node in forward_dependencies:
                next_node = forward_dependencies[node]
                if next_node in path:
                    # Found a cycle - reconstruct path (simple for now)
                    # A more robust reconstruction might be needed for complex paths
                    # For now, just report the direct cycle link involved
                    cycles_found.append(f"{node} → {next_node}")
                    return True  # Cycle detected
                if next_node not in visited and detect_cycle_recursive(next_node):
                    return True  # Propagate cycle detection

            path.remove(node)
            return False  # No cycle found from this node

        for var in forward_dependencies:
            if var not in visited:
                detect_cycle_recursive(var)

        if cycles_found:
            # Remove duplicates and format
            unique_cycles = sorted(set(cycles_found))
            cycle_descriptions = "; ".join(unique_cycles)
            errors.append(
                ValidationError(
                    f"Circular forward dependencies detected: {cycle_descriptions}. Forward references must form a chain (DAG).",
                    "/user/goals",  # General error path for cycles
                ),
            )

        return errors

    def _validate_chatbot_section(self, chatbot: dict) -> list[ValidationError]:
        """Validate the 'chatbot' section configuration.

        Args:
            chatbot: The 'chatbot' section dictionary from the YAML data.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # Validate is_starter (required field checked elsewhere)
        if "is_starter" in chatbot:
            is_starter = chatbot["is_starter"]
            if not isinstance(is_starter, bool):
                errors.append(
                    ValidationError(
                        f"is_starter must be a boolean (true or false), got {type(is_starter).__name__}",
                        "/chatbot/is_starter",
                    ),
                )

        # Validate fallback (required field checked elsewhere)
        if "fallback" in chatbot:
            fallback = chatbot["fallback"]
            if not isinstance(fallback, str):
                errors.append(
                    ValidationError(f"Fallback must be a string, got {type(fallback).__name__}", "/chatbot/fallback")
                )
            elif fallback.strip() == "":
                errors.append(ValidationError("Fallback string cannot be empty", "/chatbot/fallback"))

        # Validate output (required field checked elsewhere)
        if "output" in chatbot:
            if not isinstance(chatbot["output"], list):
                errors.append(ValidationError("Output must be a list", "/chatbot/output"))
            else:
                for i, output_item in enumerate(chatbot["output"]):
                    path = f"/chatbot/output/{i}"
                    if not isinstance(output_item, dict):
                        errors.append(
                            ValidationError(
                                f"Each output item must be a dictionary, got {type(output_item).__name__}", path
                            )
                        )
                        continue

                    if len(output_item) != 1:
                        errors.append(
                            ValidationError("Each output item must have exactly one key (the output name)", path)
                        )
                        continue

                    output_name = next(iter(output_item.keys()))
                    output_def = output_item[output_name]
                    output_path = f"{path}/{output_name}"

                    if not isinstance(output_def, dict):
                        errors.append(
                            ValidationError(f"Output definition for '{output_name}' must be a dictionary", output_path)
                        )
                        continue

                    # Check required fields within output definition
                    for field in ["type", "description"]:
                        if field not in output_def:
                            errors.append(ValidationError(f"Output '{output_name}' must have a '{field}'", output_path))

                    # Validate output type
                    if "type" in output_def:
                        output_type = output_def["type"]
                        if output_type not in self.valid_output_types:
                            errors.append(
                                ValidationError(
                                    f"Invalid output type '{output_type}'. Must be one of: {', '.join(self.valid_output_types)}",
                                    f"{output_path}/type",
                                ),
                            )

                    # Validate description is a non-empty string
                    if "description" in output_def:
                        desc = output_def["description"]
                        if not isinstance(desc, str) or desc.strip() == "":
                            errors.append(
                                ValidationError(
                                    "Description must be a non-empty string",
                                    f"{output_path}/description",
                                ),
                            )
                    # Check for unexpected keys in output definition
                    allowed_keys = {"type", "description"}
                    for key in output_def:
                        if key not in allowed_keys:
                            errors.append(
                                ValidationError(
                                    f"Unexpected key '{key}' in output definition '{output_name}'",
                                    f"{output_path}/{key}",
                                )
                            )

        return errors

    def _validate_conversation_section(self, conversation: dict) -> list[ValidationError]:
        """Validate the 'conversation' section configuration.

        Args:
            conversation: The 'conversation' section dictionary from the YAML data.

        Returns:
            A list of ValidationError objects.
        """
        errors = []

        # Validate 'number' field (required field checked elsewhere)
        if "number" in conversation:
            errors.extend(self._validate_conversation_number(conversation["number"]))

        # Validate 'max_cost' field (optional)
        if "max_cost" in conversation:
            errors.extend(self._validate_conversation_max_cost(conversation["max_cost"]))

        # Validate 'goal_style' field (required field checked elsewhere)
        if "goal_style" in conversation:
            errors.extend(self._validate_conversation_goal_style(conversation["goal_style"]))

        # Validate 'interaction_style' field (required field checked elsewhere)
        if "interaction_style" in conversation:
            errors.extend(self._validate_conversation_interaction_style(conversation["interaction_style"]))

        return errors

    def _validate_conversation_number(self, num_val: any) -> list[ValidationError]:
        """Validate the 'conversation.number' field.

        Args:
            num_val: The value of the 'conversation.number' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        path = "/conversation/number"
        allowed_msg = "Number must be a positive integer, 'all_combinations', or sample(0.0-1.0)"

        if isinstance(num_val, int):
            if num_val <= 0:
                errors.append(ValidationError("Number of conversations must be a positive integer", path))
        elif isinstance(num_val, str):
            if num_val == "all_combinations":
                pass  # Valid
            elif num_val.startswith("sample(") and num_val.endswith(")"):
                sample_param = num_val[len("sample(") : -1].strip()
                try:
                    sample_value = float(sample_param)
                    # Allow 0 exclusively, but not > 1
                    if not (0 < sample_value <= 1):
                        errors.append(
                            ValidationError("Sample value must be between 0 (exclusive) and 1 (inclusive)", path)
                        )
                except ValueError:
                    errors.append(
                        ValidationError("Invalid sample value, must be a decimal number (e.g., sample(0.5))", path)
                    )
            else:
                errors.append(ValidationError(allowed_msg, path))
        else:
            errors.append(ValidationError(allowed_msg, path))
        return errors

    def _validate_conversation_max_cost(self, cost_val: any) -> list[ValidationError]:
        """Validate the optional 'conversation.max_cost' field.

        Args:
            cost_val: The value of the 'conversation.max_cost' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        path = "/conversation/max_cost"
        if not isinstance(cost_val, (int, float)) or cost_val <= 0:
            errors.append(ValidationError(f"Max cost must be a positive number, got '{cost_val}'", path))
        return errors

    def _validate_conversation_goal_style(self, goal_style: any) -> list[ValidationError]:
        """Validate the 'conversation.goal_style' field.

        Args:
            goal_style: The value of the 'conversation.goal_style' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        path = "/conversation/goal_style"

        if isinstance(goal_style, str):
            if goal_style != "default":
                errors.append(ValidationError("When goal_style is a string, it must be 'default'", path))
        elif isinstance(goal_style, dict):
            # Check for valid goal style option keys
            for key in goal_style:
                if key not in self.valid_goal_styles:
                    errors.append(
                        ValidationError(
                            f"Invalid goal_style option: '{key}'. Valid options: {', '.join(self.valid_goal_styles)}",
                            f"{path}/{key}",
                        ),
                    )

            # Validate specific options if present and valid key
            if "steps" in goal_style:
                steps = goal_style["steps"]
                if not isinstance(steps, int) or steps <= 0:
                    errors.append(ValidationError("Steps must be a positive integer", f"{path}/steps"))

            if "random_steps" in goal_style:
                random_steps = goal_style["random_steps"]
                if not isinstance(random_steps, int) or random_steps <= 0:
                    errors.append(ValidationError("Random steps must be a positive integer", f"{path}/random_steps"))
                elif random_steps > 20:  # Example limit
                    errors.append(ValidationError("Random steps cannot exceed 20", f"{path}/random_steps"))

            if "all_answered" in goal_style:
                errors.extend(
                    self._validate_goal_style_all_answered(goal_style["all_answered"], f"{path}/all_answered")
                )

            if "max_cost" in goal_style:
                cost = goal_style["max_cost"]
                if not isinstance(cost, (int, float)) or cost <= 0:
                    errors.append(ValidationError("Goal style max_cost must be a positive number", f"{path}/max_cost"))

        else:
            errors.append(
                ValidationError(f"Goal style must be 'default' or a dictionary, got {type(goal_style).__name__}", path)
            )

        return errors

    def _validate_goal_style_all_answered(self, all_answered: any, path: str) -> list[ValidationError]:
        """Validate the 'all_answered' sub-field within 'goal_style'.

        Args:
            all_answered: The value of the 'all_answered' field.
            path: The JSON path to the 'all_answered' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if isinstance(all_answered, bool):
            pass  # Boolean value is fine
        elif isinstance(all_answered, dict):
            allowed_keys = {"export", "limit"}
            for key in all_answered:
                if key not in allowed_keys:
                    errors.append(
                        ValidationError(
                            f"Unexpected key '{key}' in all_answered. Allowed keys: {', '.join(allowed_keys)}",
                            f"{path}/{key}",
                        )
                    )

            # Validate export field if present (optional)
            if "export" in all_answered and not isinstance(all_answered["export"], bool):
                errors.append(
                    ValidationError(
                        f"Export field must be a boolean, got {type(all_answered['export']).__name__}", f"{path}/export"
                    )
                )

            # Validate limit field if present (optional)
            if "limit" in all_answered:
                limit = all_answered["limit"]
                if not isinstance(limit, int) or limit <= 0:
                    errors.append(ValidationError("Limit must be a positive integer", f"{path}/limit"))
        else:
            errors.append(
                ValidationError(
                    f"all_answered must be a boolean or a dictionary, got {type(all_answered).__name__}", path
                )
            )
        return errors

    def _validate_conversation_interaction_style(self, interaction_style: any) -> list[ValidationError]:
        """Validate the 'conversation.interaction_style' field.

        Args:
            interaction_style: The value of the 'conversation.interaction_style' field.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        path = "/conversation/interaction_style"

        if not isinstance(interaction_style, list):
            errors.append(
                ValidationError(f"Interaction style must be a list, got {type(interaction_style).__name__}", path)
            )
            return errors  # Stop validation if not a list

        if not interaction_style:
            errors.append(ValidationError("Interaction style list cannot be empty", path))

        for i, style in enumerate(interaction_style):
            item_path = f"{path}/{i}"
            if isinstance(style, str):
                if style == "change language":  # This specific string requires dictionary format
                    errors.append(
                        ValidationError(
                            "'change language' style must be a dictionary with a list of languages (e.g., {'change language': ['Spanish', 'French']})",
                            item_path,
                        )
                    )
                elif style not in self.valid_interaction_styles:
                    errors.append(
                        ValidationError(
                            f"Invalid interaction style string: '{style}'. Must be one of: {', '.join(self.valid_interaction_styles)}",
                            item_path,
                        ),
                    )
            elif isinstance(style, dict):
                errors.extend(self._validate_interaction_style_dict(style, item_path))
            else:
                errors.append(
                    ValidationError(
                        f"Interaction style item must be a string or dictionary, got {type(style).__name__}", item_path
                    )
                )
        return errors

    def _validate_interaction_style_dict(self, style_dict: dict, path: str) -> list[ValidationError]:
        """Validate a dictionary item within the 'interaction_style' list.

        Args:
            style_dict: The dictionary item from the list.
            path: The JSON path to this dictionary item.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if len(style_dict) != 1:
            errors.append(
                ValidationError(
                    "Each complex style dictionary must have exactly one key ('random' or 'change language')", path
                )
            )
            return errors  # Cannot proceed if structure is wrong

        style_type = next(iter(style_dict.keys()))
        style_value = style_dict[style_type]
        type_path = f"{path}/{style_type}"

        if style_type == "random":
            errors.extend(self._validate_interaction_style_random(style_value, type_path))
        elif style_type == "change language":
            errors.extend(self._validate_interaction_style_change_language(style_value, type_path))
        else:
            errors.append(
                ValidationError(
                    f"Invalid complex style key: '{style_type}'. Must be 'random' or 'change language'", path
                )
            )

        return errors

    def _validate_interaction_style_random(self, random_list: any, path: str) -> list[ValidationError]:
        """Validate the list associated with the 'random' interaction style.

        Args:
            random_list: The value associated with the 'random' key.
            path: The JSON path to the 'random' list.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if not isinstance(random_list, list):
            errors.append(
                ValidationError(
                    f"Random style value must be a list of other styles, got {type(random_list).__name__}", path
                )
            )
            return errors
        if not random_list:
            errors.append(ValidationError("Random style list cannot be empty", path))

        # Check each style option within the random list
        for j, random_style in enumerate(random_list):
            item_path = f"{path}/{j}"
            if isinstance(random_style, str):
                if random_style == "change language":  # Cannot be string here
                    errors.append(
                        ValidationError("'change language' within random list must be a dictionary", item_path)
                    )
                elif random_style == "random":  # Cannot nest random
                    errors.append(
                        ValidationError("Cannot nest 'random' style within another 'random' style", item_path)
                    )
                elif random_style not in self.valid_interaction_styles:
                    errors.append(ValidationError(f"Invalid style in random list: '{random_style}'", item_path))
            elif isinstance(random_style, dict):
                # Only 'change language' is allowed as a nested dictionary
                if len(random_style) != 1 or "change language" not in random_style:
                    errors.append(
                        ValidationError(
                            "Only 'change language' dictionary is allowed inside a 'random' list", item_path
                        )
                    )
                else:
                    # Validate the nested 'change language' list
                    errors.extend(
                        self._validate_interaction_style_change_language(
                            random_style["change language"], f"{item_path}/change language"
                        )
                    )
            else:
                errors.append(
                    ValidationError(
                        f"Random style items must be strings or a 'change language' dictionary, got {type(random_style).__name__}",
                        item_path,
                    )
                )
        return errors

    def _validate_interaction_style_change_language(self, lang_list: any, path: str) -> list[ValidationError]:
        """Validate the list associated with the 'change language' interaction style.

        Args:
            lang_list: The value associated with the 'change language' key.
            path: The JSON path to the 'change language' list.

        Returns:
            A list of ValidationError objects.
        """
        errors = []
        if not isinstance(lang_list, list):
            errors.append(
                ValidationError(
                    f"Change language value must be a list of languages, got {type(lang_list).__name__}", path
                )
            )
            return errors
        if not lang_list:
            errors.append(ValidationError("Change language list cannot be empty", path))

        # Check each language is a non-empty string and exists in the known languages list
        for j, language in enumerate(lang_list):
            item_path = f"{path}/{j}"
            if not isinstance(language, str):
                errors.append(ValidationError(f"Language must be a string, got {type(language).__name__}", item_path))
            elif language.strip() == "":
                errors.append(ValidationError("Language string cannot be empty", item_path))
            elif language not in languages:  # Validate against the known list
                # Provide suggestions if possible (e.g., case difference) - simple check for now
                suggestion = ""
                lower_lang = language.lower()
                matches = [language_item for language_item in languages if language_item.lower() == lower_lang]
                if matches:
                    suggestion = f". Did you mean '{matches[0]}'?"
                errors.append(
                    ValidationError(
                        f"Invalid language '{language}'. Not found in the predefined list{suggestion}", item_path
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
