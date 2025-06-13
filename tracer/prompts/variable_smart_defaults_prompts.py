"""Prompts for generating smart default options for variables."""


def get_date_variable_prompt(variable_name: str, context_preview: str, language: str = "English") -> str:
    """Get prompt for generating date variable options.

    Args:
        variable_name: Name of the variable without {{ }}
        context_preview: Preview of the profile goals context
        language: Language to generate options in

    Returns:
        Formatted prompt string
    """
    return f"""
Generate a list of 4-6 realistic date options in {language} that a user would actually select when scheduling an appointment.
These must be SPECIFIC DATE VALUES, not descriptions or labels.

Examples of GOOD date options:
- "Tomorrow"
- "Monday"
- "Next Friday"
- "December 15"
- "2024-01-20"

Examples of BAD options (do NOT include):
- "The date"
- "Your preferred date"
- "Available dates"
- "Date selection"

The variable is named '{variable_name}' and appears in this context:
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable dates:
["option1", "option2", "option3", ...]
"""


def get_time_variable_prompt(variable_name: str, context_preview: str, language: str = "English") -> str:
    """Get prompt for generating time variable options.

    Args:
        variable_name: Name of the variable without {{ }}
        context_preview: Preview of the profile goals context
        language: Language to generate options in

    Returns:
        Formatted prompt string
    """
    return f"""
Generate a list of 4-6 realistic time options in {language} that a user would actually select when scheduling an appointment.
These must be SPECIFIC TIME VALUES, not descriptions or labels.

Examples of GOOD time options:
- "9:00 AM"
- "2:30 PM"
- "14:00"
- "Morning"
- "Afternoon"

Examples of BAD options (do NOT include):
- "The time"
- "Your preferred time"
- "Available times"
- "Time selection"

The variable is named '{variable_name}' and appears in this context:
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable times:
["option1", "option2", "option3", ...]
"""


def get_type_variable_prompt(
    variable_name: str, base_concept: str, context_preview: str, language: str = "English"
) -> str:
    """Get prompt for generating type/category variable options.

    Args:
        variable_name: Name of the variable without {{ }}
        base_concept: The base concept extracted from the variable name
        context_preview: Preview of the profile goals context
        language: Language to generate options in

    Returns:
        Formatted prompt string
    """
    return f"""
Generate a list of 4-6 realistic types/categories of {base_concept} in {language}.
These must be SPECIFIC TYPE NAMES, not descriptions or labels.

Examples of GOOD type options:
- "Basic"
- "Premium"
- "Standard"
- "Express"

Examples of BAD options (do NOT include):
- "The {base_concept} type"
- "Available {base_concept}s"
- "{base_concept.title()} categories"

The variable is named '{variable_name}' and appears in this context:
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable types:
["option1", "option2", "option3", ...]
"""


def get_number_variable_prompt(variable_name: str, context_preview: str, language: str = "English") -> str:
    """Get prompt for generating number/quantity variable options.

    Args:
        variable_name: Name of the variable without {{ }}
        context_preview: Preview of the profile goals context
        language: Language to generate options in

    Returns:
        Formatted prompt string
    """
    return f"""
Generate a list of 4-6 realistic numeric quantities in {language}.
These must be SPECIFIC NUMBERS, not descriptions.

Examples of GOOD quantity options:
- "1"
- "2"
- "5"
- "One"
- "Two"

Examples of BAD options (do NOT include):
- "The quantity"
- "Number of items"
- "Amount needed"

The variable is named '{variable_name}' and appears in this context:
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable quantities:
["option1", "option2", "option3", ...]
"""


def get_generic_variable_prompt(variable_name: str, context_preview: str, language: str = "English") -> str:
    """Get prompt for generating generic variable options.

    Args:
        variable_name: Name of the variable without {{ }}
        context_preview: Preview of the profile goals context
        language: Language to generate options in

    Returns:
        Formatted prompt string
    """
    return f"""
Generate a list of 4-6 realistic options for a variable named '{variable_name}' in {language}.
These must be ACTUAL VALUES a user would select or input, NOT descriptions or labels.

Examples of GOOD options:
- Specific names, values, or choices
- Concrete options a user can select

Examples of BAD options (do NOT include):
- "The {variable_name}"
- "Your {variable_name}"
- "Available {variable_name}s"
- Descriptions about the variable

The variable appears in this context:
{context_preview}...

Output ONLY a JSON list of strings representing actual selectable values:
["option1", "option2", "option3", ...]
"""
