UNCERTAINTY_WORDS = {
    "maybe": 0.75,
    "perhaps": 0.75,
    "possibly": 0.75,
    "might": 0.75,
    "could": 0.80,
    "probably": 0.85,
    "possibly": 0.75,
    "seems": 0.85,
    "seemingly": 0.85,
    "guess": 0.80,
    "think": 0.90,
    "believe": 0.90,
}


CERTAINTY_WORDS = {
    "definitely": 1.15,
    "certainly": 1.15,
    "clearly": 1.10,
    "undoubtedly": 1.20,
    "surely": 1.10,
}


def get_certainty_modifier(tokens: list[str]) -> float:
    """
    Calculate a confidence modifier from certainty/uncertainty words.

    Values below 1.0 reduce confidence.
    Values above 1.0 increase confidence.
    """

    modifier = 1.0

    for token in tokens:
        word = token.lower()

        if word in UNCERTAINTY_WORDS:
            modifier *= UNCERTAINTY_WORDS[word]

        elif word in CERTAINTY_WORDS:
            modifier *= CERTAINTY_WORDS[word]

    return modifier