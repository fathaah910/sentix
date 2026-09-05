"""
Sentiment reversal detection for Sentix.

Detects constructions where an initially negative
experience/state is followed by a positive outcome.

Examples:

    The terrible wait was completely worth it.
    The bad experience ended beautifully.
    The awful process turned out great.

The rule is intentionally conservative.
"""

from ..normalize import normalize_word


# ---------------------------------------------------------
# Positive outcome phrases
#
# IMPORTANT:
# These phrases use normalized word forms.
#
# "ended" -> "end" in Sentix normalization.
# ---------------------------------------------------------

POSITIVE_OUTCOME_PHRASES = {
    ("worth", "it"),
    ("worth", "the", "money"),
    ("worth", "the", "time"),

    ("end", "well"),
    ("end", "great"),
    ("end", "amazing"),
    ("end", "beautiful"),
    ("end", "beautifully"),

    ("turn", "out", "great"),
    ("turn", "out", "amazing"),
    ("turn", "out", "beautiful"),
    ("turn", "out", "beautifully"),
    ("turn", "out", "well"),

    ("work", "out", "great"),
    ("work", "out", "amazing"),
    ("work", "out", "beautiful"),
    ("work", "out", "beautifully"),
    ("work", "out", "well"),
}


# ---------------------------------------------------------
# Positive outcome words
# ---------------------------------------------------------

POSITIVE_OUTCOME_WORDS = {
    "beautiful",
    "beautifully",
    "great",
    "amazing",
    "excellent",
    "wonderful",
    "perfect",
    "successful",
    "successfully",
    "well",
}


# ---------------------------------------------------------
# Negative experience/state words
# ---------------------------------------------------------

NEGATIVE_EXPERIENCE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "disappointing",
    "ugly",
    "hate",
    "dislike",
}


def _normalized_tokens(
    tokens: list[str],
) -> list[str]:
    """
    Normalize tokens while preserving order.
    """

    return [
        normalize_word(token)
        for token in tokens
    ]


def _find_positive_outcome(
    tokens: list[str],
) -> list[tuple[int, int, str]]:
    """
    Find known positive outcome constructions.

    Returns:

        [
            (start_index, end_index, phrase)
        ]

    end_index is exclusive.
    """

    normalized = _normalized_tokens(
        tokens
    )

    matches: list[
        tuple[int, int, str]
    ] = []

    # ---------------------------------------------------------
    # Exact multi-word outcome phrases
    # ---------------------------------------------------------

    for phrase in POSITIVE_OUTCOME_PHRASES:

        phrase_length = len(phrase)

        if phrase_length > len(normalized):
            continue

        for index in range(
            len(normalized)
            - phrase_length
            + 1
        ):

            window = tuple(
                normalized[
                    index:
                    index + phrase_length
                ]
            )

            if window == phrase:

                matches.append(
                    (
                        index,
                        index + phrase_length,
                        " ".join(phrase),
                    )
                )

    return matches


def _find_negative_experience(
    tokens: list[str],
) -> list[int]:
    """
    Find negative experience/state words.
    """

    normalized = _normalized_tokens(
        tokens
    )

    return [
        index
        for index, token in enumerate(
            normalized
        )
        if token in NEGATIVE_EXPERIENCE_WORDS
    ]


def detect_reversal(
    tokens: list[str],
) -> dict[str, object]:
    """
    Detect negative-experience -> positive-outcome
    sentiment reversal.

    Returns:

        {
            "has_reversal": bool,
            "negative_indexes": [...],
            "outcome_indexes": [...],
            "positive_score": float,
            "reason": str | None,
        }
    """

    negative_indexes = (
        _find_negative_experience(
            tokens
        )
    )

    outcome_matches = (
        _find_positive_outcome(
            tokens
        )
    )

    # ---------------------------------------------------------
    # Nothing to reverse
    # ---------------------------------------------------------

    if not negative_indexes:
        return {
            "has_reversal": False,
            "negative_indexes": [],
            "outcome_indexes": [],
            "positive_score": 0.0,
            "reason": None,
        }

    if not outcome_matches:
        return {
            "has_reversal": False,
            "negative_indexes": [],
            "outcome_indexes": [],
            "positive_score": 0.0,
            "reason": None,
        }

    # ---------------------------------------------------------
    # Only accept an outcome that occurs after
    # a negative experience.
    # ---------------------------------------------------------

    valid_matches = []

    for (
        start,
        end,
        phrase,
    ) in outcome_matches:

        has_negative_before = any(
            negative_index < start
            for negative_index
            in negative_indexes
        )

        if has_negative_before:

            valid_matches.append(
                (
                    start,
                    end,
                    phrase,
                )
            )

    if not valid_matches:
        return {
            "has_reversal": False,
            "negative_indexes": [],
            "outcome_indexes": [],
            "positive_score": 0.0,
            "reason": None,
        }

    # ---------------------------------------------------------
    # Collect outcome indexes
    # ---------------------------------------------------------

    outcome_indexes: list[int] = []

    for (
        start,
        end,
        _,
    ) in valid_matches:

        outcome_indexes.extend(
            range(
                start,
                end,
            )
        )

    outcome_indexes = sorted(
        set(outcome_indexes)
    )

    # ---------------------------------------------------------
    # Strong contextual positive evidence.
    # ---------------------------------------------------------

    positive_score = 3.0

    return {
        "has_reversal": True,
        "negative_indexes": sorted(
            set(negative_indexes)
        ),
        "outcome_indexes": outcome_indexes,
        "positive_score": positive_score,
        "reason": (
            "negative experience is followed "
            "by a positive outcome"
        ),
    }