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

# Conversation loop constants
MIN_EXPLORER_RESPONSE_LENGTH = 10
CONTEXT_MESSAGES_COUNT = 4
MIN_CORRECTED_MESSAGE_LENGTH = 5

# Chatbot communication resilience
CHATBOT_MAX_RETRIES = 3
CHATBOT_RETRY_BACKOFF_SECONDS = 1.0
CHATBOT_TOKEN_BUCKET_CAPACITY = 1
CHATBOT_TOKEN_BUCKET_REFILL_RATE_PER_SECOND = 0.25
CHATBOT_RETRY_BACKOFF_MAX_SECONDS = 30.0
CHATBOT_MIN_INTER_MESSAGE_GAP_SECONDS = 1.0
CHATBOT_TYPING_DELAY_PER_CHARACTER_SECONDS = 0.005
CHATBOT_TYPING_DELAY_JITTER_SECONDS = 1.5
CHATBOT_READING_DELAY_PER_CHARACTER_SECONDS = 0.003
CHATBOT_SESSION_COOLDOWN_BASE_SECONDS = 45.0
CHATBOT_SESSION_COOLDOWN_JITTER_SECONDS = 15.0
CHATBOT_SESSION_ISSUE_COOLDOWN_BASE_SECONDS = 180.0
CHATBOT_SESSION_ISSUE_COOLDOWN_JITTER_SECONDS = 30.0

# JSON parsing constants
MIN_PRINTABLE_ASCII_CODE = 32
