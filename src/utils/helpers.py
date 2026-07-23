
import re


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()