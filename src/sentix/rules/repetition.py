import re


def has_repeated_characters(word: str) -> bool:
    return bool(re.search(r"(.)\1{2,}", word))

def normalize_repeated_characters(word: str) -> str:
    """
    Reduce characters repeated three or more times
    to a single character.

    Example:
        loooove -> love
        goooood -> god
    """
    return re.sub(r"(.)\1{2,}", r"\1", word)