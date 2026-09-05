"""
Negation scope detection for Sentix.

This module handles simple and nested negation.

Examples:

    This is not good.
    I don't think this is good.
    I don't think this is not good.
    I can't say that I don't like this.
    It's not that I hate this.

The implementation is intentionally rule-based and
conservative. It is not a full dependency parser.
"""


NEGATIONS = {
    "not",
    "no",
    "never",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "wasn't",
    "aren't",
    "weren't",
    "can't",
    "cannot",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
}


NEGATION_TERMINATORS = {
    ".",
    "!",
    "?",
    ",",
    ";",
    ":",
}


CLAUSE_VERBS = {
    "think",
    "believe",
    "say",
    "feel",
    "assume",
    "suppose",
    "guess",
}


def is_negation(token: str) -> bool:
    """
    Check whether a token is a negation.
    """

    return token.lower() in NEGATIONS


def _previous_negation_count(
    tokens: list[str],
    index: int,
    window: int = 8,
) -> int:
    """
    Count negations affecting a sentiment word.

    The search stops at strong punctuation.

    Example:

        I don't think this is not good.

        good -> 2 negations
    """

    start = max(
        0,
        index - window,
    )

    count = 0

    for i in range(
        index - 1,
        start - 1,
        -1,
    ):

        token = tokens[i]

        if token in NEGATION_TERMINATORS:
            break

        if is_negation(token):
            count += 1

    return count


def get_negation_count(
    tokens: list[str],
    index: int,
    window: int = 8,
) -> int:
    """
    Return the number of active negations before
    a sentiment-bearing token.

    Example:

        I don't think this is not good.

        good -> 2
    """

    return _previous_negation_count(
        tokens,
        index,
        window,
    )


def get_negation_modifier(
    tokens: list[str],
    index: int,
    window: int = 8,
) -> float:
    """
    Return the polarity modifier caused by negation.

    Odd number of negations:

        positive -> negative
        negative -> positive

    Even number of negations:

        sentiment remains unchanged.
    """

    count = get_negation_count(
        tokens,
        index,
        window,
    )

    if count % 2 == 1:
        return -1.0

    return 1.0


def has_negation_before(
    tokens: list[str],
    index: int,
    window: int = 8,
) -> bool:
    """
    Backward-compatible helper.

    Returns True when at least one negation affects
    the sentiment word.
    """

    return (
        get_negation_count(
            tokens,
            index,
            window,
        )
        > 0
    )


def _find_outer_negation(
    tokens: list[str],
    index: int,
) -> int | None:
    """
    Find an outer negation such as:

        I don't think ...
        I can't say ...
        It's not that ...

    before the sentiment clause.
    """

    start = max(
        0,
        index - 8,
    )

    for i in range(
        index - 1,
        start - 1,
        -1,
    ):

        token = tokens[i].lower()

        if token in NEGATION_TERMINATORS:
            break

        if token not in NEGATIONS:
            continue

        for j in range(
            i + 1,
            min(i + 4, len(tokens)),
        ):

            next_token = tokens[j].lower()

            if next_token in CLAUSE_VERBS:
                return i

            if tokens[j] in NEGATION_TERMINATORS:
                break

    return None


def get_negation_scope(
    tokens: list[str],
    index: int,
) -> dict[str, object]:
    """
    Analyze negation affecting a sentiment token.

    Returns:

        {
            "count": int,
            "modifier": float,
            "negated": bool,
            "nested": bool,
            "outer_negation": bool,
        }
    """

    count = get_negation_count(
        tokens,
        index,
    )

    outer_negation = (
        _find_outer_negation(
            tokens,
            index,
        )
        is not None
    )

    return {
        "count": count,
        "modifier": (
            -1.0
            if count % 2 == 1
            else 1.0
        ),
        "negated": count > 0,
        "nested": count >= 2,
        "outer_negation": outer_negation,
    }