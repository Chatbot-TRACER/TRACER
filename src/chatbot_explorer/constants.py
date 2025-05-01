"""Necessary constants for the chatbot exploration framework."""

import re

# Regex to find {{variables}}
VARIABLE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")

# Personalities to choose one for the profile
AVAILABLE_PERSONALITIES = [
    "conversational-user",
    "curious-user",
    "direct-user",
    "disorganized-user",
    "elderly-user",
    "formal-user",
    "impatient-user",
    "rude-user",
    "sarcastic-user",
    "skeptical-user",
]
