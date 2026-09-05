import re


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase tokens while preserving
    contractions and punctuation.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:\.\d+)?|[^\w\s]",
        text
    )