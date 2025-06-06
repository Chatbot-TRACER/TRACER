"""Provides utility functions for extracting YAML and JSON data from text."""

import re

MIN_POTENTIAL_YAML_LENGTH = 10


def extract_yaml(text: str) -> str:
    """Extract YAML content from LLM response text.

    Args:
        text: Text potentially containing YAML

    Returns:
        str: Extracted YAML content
    """
    # Handle LangChain message object
    if hasattr(text, "content"):
        text = text.content

    # Try common code fence patterns
    yaml_patterns = [
        r"```\s*yaml\s*(.*?)```",
        r"```\s*YAML\s*(.*?)```",
        r"```(.*?)```",
        r"`{3,}(.*?)`{3,}",
    ]

    for pattern in yaml_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            # Basic check if it looks like YAML
            if ":" in extracted and len(extracted) > MIN_POTENTIAL_YAML_LENGTH:
                return extracted

    # If no fences, check if it starts like YAML
    if "test_name:" in text or "user:" in text or "chatbot:" in text:
        # Try to strip leading non-YAML lines
        lines = text.strip().split("\n")
        while lines and not any(
            keyword in lines[0]
            for keyword in [
                "test_name:",
                "user:",
                "chatbot:",
                "llm:",
            ]
        ):
            lines.pop(0)
        return "\n".join(lines)

    # Give up and return stripped text
    return text.strip()


def extract_json_from_response(response_content: str) -> str:
    """Extract JSON content from the LLM response."""
    json_str = None
    json_patterns = [
        r"```json\s*([\s\S]+?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]+?)\s*```",  # ``` ... ```
        r"\[\s*\{.*?\}\s*\]",  # Starts with [ { and ends with } ]
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_content, re.DOTALL)
        if match:
            json_str = match.group(1) if "```" in pattern else match.group(0)
            break

    # Fallback if no pattern matched
    if not json_str:
        if response_content.strip().startswith("[") and response_content.strip().endswith("]"):
            json_str = response_content.strip()
        else:
            msg = "Could not extract JSON block from LLM response."
            raise ValueError(msg)

    return json_str
