from ..lexicon.aspects import ASPECT_LEXICON
from ..lexicon.sentiment import SENTIMENT_LEXICON
from .negation import has_negation_before
from .phrases import find_phrases
from .intensity import get_intensity_modifier


def detect_aspects(tokens: list[str]) -> dict[str, dict[str, float | str]]:
    """
    Detect aspects and return their sentiment and score.
    """

    aspects = {}

    # Find sentiment phrases first
    phrase_matches = find_phrases(tokens)

    for index, token in enumerate(tokens):

        normalized_token = token.lower()

        if normalized_token not in ASPECT_LEXICON:
            continue

        sentiment = "neutral"
        aspect_score = 0.0

        # --------------------------------
        # Check sentiment phrases
        # --------------------------------

        for start, end, phrase_score in phrase_matches:

            if start <= index:
                continue

            if start - index > 4:
                continue

            score = phrase_score

            if has_negation_before(tokens, start):
                score *= -1

            aspect_score = score

            if score > 0:
                sentiment = "positive"

            elif score < 0:
                sentiment = "negative"

            break

        # --------------------------------
        # Check individual sentiment words
        # --------------------------------

        if sentiment == "neutral":

            for next_index in range(
                index + 1,
                min(index + 4, len(tokens))
            ):

                next_token = tokens[next_index].lower()

                if next_token in SENTIMENT_LEXICON:

                    score = SENTIMENT_LEXICON[next_token]

                    # Apply intensity
                    modifier = get_intensity_modifier(
                        tokens,
                        next_index
                    )

                    score *= modifier

                    # Apply negation
                    if has_negation_before(tokens, next_index):
                        score *= -1

                    aspect_score = score

                    if score > 0:
                        sentiment = "positive"

                    elif score < 0:
                        sentiment = "negative"

                    break

        aspects[normalized_token] = {
            "sentiment": sentiment,
            "score": aspect_score,
        }

    return aspects