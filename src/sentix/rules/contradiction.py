"""
Contextual contradiction detection for Sentix.

Detects cases where positive language conflicts with
a clearly negative situation.

This is contextual contradiction detection, not a
general-purpose sarcasm detector.
"""

from ..lexicon.sentiment import SENTIMENT_LEXICON
from ..normalize import normalize_word


POSITIVE_CONTEXT_WORDS = {
    "good",
    "great",
    "amazing",
    "awesome",
    "fantastic",
    "wonderful",
    "excellent",
    "perfect",
    "beautiful",
    "love",
    "like",
    "best",
}


NEGATIVE_CONTEXT_WORDS = {
    "error",
    "errors",
    "problem",
    "problems",
    "crash",
    "crashed",
    "crashing",
    "broken",
    "breaking",
    "break",
    "stop",
    "stopped",
    "stopping",
    "failure",
    "failed",
    "fail",
    "bug",
    "bugs",
    "issue",
    "issues",
    "wrong",
    "badly",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "hate",
}


NEGATIVE_EVENT_WORDS = {
    "crash",
    "crashed",
    "crashing",
    "break",
    "breaking",
    "broken",
    "stop",
    "stopped",
    "stopping",
    "fail",
    "failed",
    "failure",
    "error",
    "errors",
    "problem",
    "problems",
    "bug",
    "bugs",
    "issue",
    "issues",
}


# -------------------------------------------------------------
# Contrast markers
# -------------------------------------------------------------
#
# A sentence containing "but" represents ordinary contrast.
#
# Example:
#
#     I love the camera, but I hate the battery.
#
# This must NOT become contextual contradiction.
#
CONTRAST_MARKERS = {
    "but",
    "however",
    "although",
    "though",
}


def _raw_word(
    token: str,
) -> str:
    """
    Remove simple punctuation from a token.
    """

    return token.lower().strip(
        ".,!?;:-"
    )


def _is_positive_word(
    token: str,
) -> bool:
    """
    Determine whether a token expresses positive sentiment.
    """

    raw = _raw_word(token)

    if raw in POSITIVE_CONTEXT_WORDS:
        return True

    normalized = normalize_word(token)

    if normalized in POSITIVE_CONTEXT_WORDS:
        return True

    if normalized in SENTIMENT_LEXICON:
        return (
            SENTIMENT_LEXICON[normalized] > 0
        )

    return False


def _is_negative_context_word(
    token: str,
) -> bool:
    """
    Determine whether a token represents negative context.
    """

    raw = _raw_word(token)

    if raw in NEGATIVE_CONTEXT_WORDS:
        return True

    normalized = normalize_word(token)

    return (
        normalized in NEGATIVE_CONTEXT_WORDS
    )


def _is_negative_event(
    token: str,
) -> bool:
    """
    Determine whether a token represents an explicit
    negative event.
    """

    raw = _raw_word(token)

    if raw in NEGATIVE_EVENT_WORDS:
        return True

    normalized = normalize_word(token)

    return (
        normalized in NEGATIVE_EVENT_WORDS
    )


def _has_contrast_marker_between(
    tokens: list[str],
    start: int,
    end: int,
) -> bool:
    """
    Return True when an explicit contrast marker occurs
    between two token positions.

    Example:

        I love the camera, but I hate the battery.
          ^ positive          ^ negative

    The "but" means this is a contrast relation rather
    than contextual contradiction.
    """

    left = min(start, end)
    right = max(start, end)

    for index in range(left + 1, right):
        word = _raw_word(tokens[index])

        if word in CONTRAST_MARKERS:
            return True

        # Handle attached forms such as:
        #
        #     buttoday
        #     butmaybe
        #
        if (
            word.startswith("but")
            and len(word) > 3
        ):
            return True

    return False


def detect_contradiction(
    tokens: list[str],
) -> dict[str, object]:
    """
    Detect contextual contradiction.

    Returns:

        {
            "has_contradiction": bool,
            "positive_indexes": [...],
            "negative_indexes": [...],
            "negative_score": float,
            "reason": str | None,
        }
    """

    positive_indexes: list[int] = []
    negative_indexes: list[int] = []

    # =========================================================
    # PATTERN 1
    #
    # Positive expression + punctuation + negative situation
    #
    # Examples:
    #
    #     Fantastic, another error.
    #     Wonderful, the app crashed.
    #     Perfect, another problem.
    #
    # IMPORTANT:
    #
    # If "but" occurs between the positive expression and
    # negative word, this is ordinary contrast and must NOT
    # be classified as contradiction.
    # =========================================================

    for index, token in enumerate(tokens):

        if not _is_positive_word(token):
            continue

        if index + 1 >= len(tokens):
            continue

        if tokens[index + 1] not in {
            ",",
            ":",
            "-",
            "!",
        }:
            continue

        context_end = min(
            len(tokens),
            index + 8,
        )

        negative_indexes_here = [
            context_index
            for context_index in range(
                index + 2,
                context_end,
            )
            if _is_negative_context_word(
                tokens[context_index]
            )
        ]

        for negative_index in negative_indexes_here:

            if _has_contrast_marker_between(
                tokens,
                index,
                negative_index,
            ):
                continue

            positive_indexes.append(index)
            negative_indexes.append(
                negative_index
            )

    # =========================================================
    # PATTERN 2
    #
    # Positive expression + negative situation separated
    # somewhere within a short context.
    #
    # Example:
    #
    #     Great, now everything is broken.
    #
    # IMPORTANT:
    #
    # A "but" relationship is handled by contrast.py,
    # not contradiction.py.
    # =========================================================

    for index, token in enumerate(tokens):

        if not _is_positive_word(token):
            continue

        context_end = min(
            len(tokens),
            index + 8,
        )

        negative_indexes_here = [
            context_index
            for context_index in range(
                index + 1,
                context_end,
            )
            if _is_negative_context_word(
                tokens[context_index]
            )
        ]

        if not negative_indexes_here:
            continue

        for negative_index in negative_indexes_here:

            if _has_contrast_marker_between(
                tokens,
                index,
                negative_index,
            ):
                continue

            has_separator = any(
                tokens[i] in {
                    ",",
                    ":",
                    ";",
                    "!",
                    "?",
                }
                for i in range(
                    index + 1,
                    negative_index + 1,
                )
            )

            if not has_separator:
                continue

            positive_indexes.append(index)
            negative_indexes.append(
                negative_index
            )

    # =========================================================
    # PATTERN 3
    #
    # Positive + how + negative
    #
    # Example:
    #
    #     Amazing how terrible this product is.
    # =========================================================

    for index, token in enumerate(tokens):

        if not _is_positive_word(token):
            continue

        context_end = min(
            len(tokens),
            index + 8,
        )

        for context_index in range(
            index + 1,
            context_end,
        ):

            if (
                tokens[context_index].lower()
                != "how"
            ):
                continue

            negative_indexes_here = [
                i
                for i in range(
                    context_index + 1,
                    context_end,
                )
                if _is_negative_context_word(
                    tokens[i]
                )
            ]

            if negative_indexes_here:

                positive_indexes.append(index)

                negative_indexes.extend(
                    negative_indexes_here
                )

    # =========================================================
    # PATTERN 4
    #
    # Positive opinion + negative event
    #
    # Example:
    #
    #     I like the fact that it keeps crashing.
    #
    # Deliberately restricted to explicit negative event
    # words and connecting constructions.
    # =========================================================

    for index, token in enumerate(tokens):

        if not _is_positive_word(token):
            continue

        context_end = min(
            len(tokens),
            index + 10,
        )

        negative_event_indexes = [
            context_index
            for context_index in range(
                index + 1,
                context_end,
            )
            if _is_negative_event(
                tokens[context_index]
            )
        ]

        if not negative_event_indexes:
            continue

        context_tokens = [
            _raw_word(tokens[i])
            for i in range(
                index + 1,
                context_end,
            )
        ]

        has_connection = any(
            word in {
                "fact",
                "that",
                "keeps",
                "keep",
                "kept",
                "continues",
                "continue",
                "continued",
                "still",
                "always",
                "constantly",
            }
            for word in context_tokens
        )

        if has_connection:

            positive_indexes.append(index)

            negative_indexes.extend(
                negative_event_indexes
            )

    # =========================================================
    # PATTERN 5
    #
    # Positive expression + "job" + negative action
    #
    # Example:
    #
    #     Yeah, great job breaking everything.
    # =========================================================

    for index, token in enumerate(tokens):

        if not _is_positive_word(token):
            continue

        context_end = min(
            len(tokens),
            index + 7,
        )

        context = [
            _raw_word(tokens[i])
            for i in range(
                index + 1,
                context_end,
            )
        ]

        if "job" not in context:
            continue

        negative_event_indexes = [
            context_index
            for context_index in range(
                index + 1,
                context_end,
            )
            if _is_negative_event(
                tokens[context_index]
            )
        ]

        if negative_event_indexes:

            positive_indexes.append(index)

            negative_indexes.extend(
                negative_event_indexes
            )

    # =========================================================
    # DEDUPLICATE
    # =========================================================

    positive_indexes = sorted(
        set(positive_indexes)
    )

    negative_indexes = sorted(
        set(negative_indexes)
    )

    has_contradiction = bool(
        positive_indexes
        and negative_indexes
    )

    if not has_contradiction:

        return {
            "has_contradiction": False,
            "positive_indexes": [],
            "negative_indexes": [],
            "negative_score": 0.0,
            "reason": None,
        }

    # Strong contextual negative signal.
    negative_score = -1.5

    return {
        "has_contradiction": True,
        "positive_indexes": positive_indexes,
        "negative_indexes": negative_indexes,
        "negative_score": negative_score,
        "reason": (
            "positive expression conflicts "
            "with negative context"
        ),
    }