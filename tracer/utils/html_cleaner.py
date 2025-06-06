"""Module for cleaning HTML responses from chatbots."""

import re

from bs4 import BeautifulSoup

from tracer.utils.logging_utils import get_logger

MAX_HTML_PATTERNS = 2

MAX_HTML_TAGS = 5


logger = get_logger()


def is_likely_html(text: str) -> bool:
    """Determines if the provided text is likely to be HTML content.

    Args:
        text: The text to analyze

    Returns:
        True if the text appears to be HTML, False otherwise
    """
    # Look for common HTML patterns
    html_patterns = [
        r"<\s*html",  # Opening html tag
        r"<\s*body",  # Opening body tag
        r"<\s*div",  # Opening div tag
        r"<\s*p\s*>",  # Paragraph tag
        r"<\s*h[1-6]",  # Header tags
        r"<\s*table",  # Table tag
        r"<\s*ul",  # Unordered list
        r"<\s*ol",  # Ordered list
        r"<\s*a\s+href",  # Anchor tag with href
        r"<\s*img\s+src",  # Image tag with src
        r"<\s*style",  # Style tag
        r"<\s*script",  # Script tag
    ]

    # Check if text contains sufficient HTML patterns
    html_pattern_count = sum(1 for pattern in html_patterns if re.search(pattern, text, re.IGNORECASE))

    # Additional check for HTML structure
    tag_count = len(re.findall(r"<[^>]+>", text))

    # Consider it HTML if it matches multiple HTML patterns or has many tags
    return html_pattern_count >= MAX_HTML_PATTERNS or tag_count >= MAX_HTML_TAGS


def clean_html_response(response: str) -> str:
    """Cleans HTML responses, it first checks if it is html.

    Args:
        response: The chatbot response, potentially containing HTML

    Returns:
        Cleaned text content if the response is HTML, otherwise returns the original response
    """
    if not response or not is_likely_html(response):
        return response

    try:
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response, "html.parser")

        # Remove script and style elements that might contain non-visible content
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get text
        text = soup.get_text(separator="\n")

        # Normalize whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

    except (ValueError, TypeError):
        logger.exception("An error occurred while cleaning HTML response.")
        return response
    else:
        return text
