"""Necessary constants for the chatbot exploration framework."""

import re

# Regular expression pattern to find {{variables}} in text
VARIABLE_PATTERN = re.compile(r"{{([^}]+)}}")

# List truncation threshold for data preview
LIST_TRUNCATION_THRESHOLD = 3

# Variable type pattern definitions supporting English and Spanish
VARIABLE_PATTERNS = {
    "date": ["date", "fecha"],
    "time": ["time", "hora"],
    "type": ["type", "tipo"],
    "number_of": ["number_of", "cantidad", "numero"],
    "price": ["price", "cost", "precio", "costo"],
}

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

# Minimum number of nodes required for deduplication
MIN_NODES_FOR_DEDUPLICATION = 2
