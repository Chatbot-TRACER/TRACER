import re


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
            if ":" in extracted and len(extracted) > 10:
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
