from ..lexicon.phrases import PHRASE_LEXICON


def find_phrases(tokens: list[str]) -> list[tuple[int, int, float]]:
    """
    Find sentiment phrases inside a token list.

    Returns:
        A list containing:
        (start_index, end_index, score)
    """

    matches = []

    for phrase, score in PHRASE_LEXICON.items():
        phrase_tokens = phrase.split()
        phrase_length = len(phrase_tokens)

        for i in range(len(tokens) - phrase_length + 1):
            window = [
                token.lower()
                for token in tokens[i:i + phrase_length]
            ]

            if window == phrase_tokens:
                matches.append(
                    (i, i + phrase_length, score)
                )

    return matches