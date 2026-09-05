from ..lexicon.aspects import ASPECT_LEXICON
from ..lexicon.sentiment import SENTIMENT_LEXICON
from .negation import has_negation_before
from .intensity import get_intensity_modifier
from ..normalize import normalize_word


SENTENCE_TERMINATORS = {".", "!", "?"}

CLAUSE_TERMINATORS = {
    ",",
    ";",
    ":",
}


def get_sentence_index(
    tokens: list[str],
    index: int,
) -> int:
    """
    Return the sentence number containing a token.

    Example:

        I love camera. I hate battery.
        -------------  ---------------
        sentence 0     sentence 1
    """

    sentence_index = 0

    for i in range(index):
        if tokens[i] in SENTENCE_TERMINATORS:
            sentence_index += 1

    return sentence_index


def split_sentences(
    tokens: list[str],
) -> list[list[str]]:
    """
    Split tokens into sentences.
    """

    sentences = []
    current = []

    for token in tokens:
        current.append(token)

        if token in SENTENCE_TERMINATORS:
            sentences.append(current)
            current = []

    if current:
        sentences.append(current)

    return sentences


def find_aspect_sentiments(
    tokens: list[str],
) -> list[dict[str, object]]:
    """
    Find sentiment associated with explicit aspects.

    Example:

        I don't like the camera. I love the screen.

    Produces approximately:

        camera -> negative
        screen -> positive
    """

    results = []

    for index, token in enumerate(tokens):

        aspect = token.lower()

        if aspect not in ASPECT_LEXICON:
            continue

        best_score = 0.0
        best_distance = None

        start = max(0, index - 4)
        end = min(len(tokens), index + 5)

        for sentiment_index in range(start, end):

            if sentiment_index == index:
                continue

            sentiment_token = normalize_word(
                tokens[sentiment_index]
            )
 
            if sentiment_token not in SENTIMENT_LEXICON:
                continue

            left = min(index, sentiment_index)
            right = max(index, sentiment_index)

            # Never connect sentiment across a sentence boundary.
            crossed_sentence = any(
                token in SENTENCE_TERMINATORS
                for token in tokens[left:right]
            )

            if crossed_sentence:
                continue

            # Do not connect across a clause boundary.
            crossed_clause = any(
                token in CLAUSE_TERMINATORS
                for token in tokens[left:right]
            )

            if crossed_clause:
                continue

            score = SENTIMENT_LEXICON[sentiment_token]

            modifier = get_intensity_modifier(
                tokens,
                sentiment_index
            )

            score *= modifier

            if has_negation_before(
                tokens,
                sentiment_index
            ):
                score *= -1

            distance = abs(
                sentiment_index - index
            )

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_score = score
                best_distance = distance

        if best_distance is not None:

            if best_score > 0:
                sentiment = "positive"

            elif best_score < 0:
                sentiment = "negative"

            else:
                sentiment = "neutral"

            results.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "score": best_score,
                    "sentence_index": get_sentence_index(
                        tokens,
                        index
                    ),
                }
            )

    return results


def analyze_context(
    tokens: list[str],
) -> dict[str, object]:
    """
    Analyze sentiment targets and their sentence context.

    Different explicit targets in different sentences can
    indicate mixed sentiment.

    Different targets inside the same sentence are left to
    the normal overall sentiment scoring logic.
    """

    aspect_sentiments = find_aspect_sentiments(tokens)

    positive_aspects = [
        item
        for item in aspect_sentiments
        if item["sentiment"] == "positive"
    ]

    negative_aspects = [
        item
        for item in aspect_sentiments
        if item["sentiment"] == "negative"
    ]

    positive_targets = {
        item["aspect"]
        for item in positive_aspects
    }

    negative_targets = {
        item["aspect"]
        for item in negative_aspects
    }

    positive_sentence_indices = {
        item["sentence_index"]
        for item in positive_aspects
    }

    negative_sentence_indices = {
        item["sentence_index"]
        for item in negative_aspects
    }

    # We only use the target-aware mixed rule when positive
    # and negative explicit targets occur in different sentences.
    different_sentence_targets = any(
        positive_sentence != negative_sentence
        for positive_sentence in positive_sentence_indices
        for negative_sentence in negative_sentence_indices
    )

    return {
        "aspect_sentiments": aspect_sentiments,
        "positive_targets": positive_targets,
        "negative_targets": negative_targets,
        "different_targets": bool(
            positive_targets
            and negative_targets
            and positive_targets != negative_targets
        ),
        "different_sentence_targets": (
            bool(positive_targets)
            and bool(negative_targets)
            and positive_targets != negative_targets
            and different_sentence_targets
        ),
    }