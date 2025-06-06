"""Prompts for building and fixing YAML configuration files."""


def get_yaml_fix_prompt(error_messages: str, yaml_content: str) -> str:
    """Generate the prompt for asking the LLM to fix YAML validation errors."""
    return (
        "You are an AI assistant specialized in correcting YAML configuration files.\n"
        "Based ONLY on the following validation errors, please fix the provided YAML content.\n"
        "Your response MUST contain ONLY the complete, corrected YAML content.\n"
        "Enclose the corrected YAML within triple backticks (```yaml ... ```).\n"
        "Do NOT include any explanations, apologies, introductions, or conclusions outside the YAML block.\n"
        "Ensure the output is well-formed YAML and directly addresses the errors listed.\n\n"
        f"Errors to fix:\n{error_messages}\n\n"
        "Original YAML to fix:\n"
        f"```yaml\n{yaml_content}\n```\n\n"
        "Corrected YAML:"
    )
