"""
Conditional and counterfactual sentiment detection for Sentix.
"""

import re


CONDITIONAL_MARKERS = {
    "if",
}


HYPOTHETICAL_MARKERS = {
    "would",
    "could",
    "might",
    "were",
}


IMPROVEMENT_WORDS = {
    "better",
    "best",
    "amazing",
    "excellent",
    "great",
    "cheaper",
    "faster",
    "easier",
    "stronger",
    "higher",
    "lower",
    "longer",
    "shorter",
    "bigger",
    "smaller",
    "nicer",
    "beautiful",
}


RESULT_WORDS = {
    "love",
    "like",
    "enjoy",
    "recommend",
    "buy",
    "choose",
    "prefer",
}


def _split_attached_if(
    tokens: list[str],
) -> list[str]:
    """
    Split an 'if' attached to the previous word.

    Example:

        thisif

    becomes:

        this, if
    """

    result: list[str] = []

    for token in tokens:

        word = token.lower()

        if (
            word.endswith("if")
            and word != "if"
            and len(word) > 2
        ):
            result.append(token[:-2])
            result.append("if")
        else:
            result.append(token)

    return result


def find_conditional_markers(
    tokens: list[str],
) -> list[int]:
    """
    Return indexes of conditional markers.
    """

    logical_tokens = _split_attached_if(tokens)

    return [
        index
        for index, token in enumerate(logical_tokens)
        if token.lower() == "if"
    ]


def find_hypothetical_markers(
    tokens: list[str],
) -> list[int]:
    """
    Return indexes of hypothetical markers.
    """

    logical_tokens = _split_attached_if(tokens)

    return [
        index
        for index, token in enumerate(logical_tokens)
        if token.lower() in HYPOTHETICAL_MARKERS
    ]


def is_conditional_context(
    tokens: list[str],
    index: int,
) -> bool:
    """
    Determine whether a sentiment token belongs to a
    conditional/hypothetical construction.

    Important:

        I would love this if the screen were better.

    'love' is the result of the hypothetical condition,
    so it should NOT be removed from sentiment scoring.

    The condition itself is handled separately by
    find_counterfactual_condition().
    """

    logical_tokens = _split_attached_if(tokens)

    # ---------------------------------------------------------
    # Map original index -> logical index.
    # ---------------------------------------------------------

    logical_index = 0

    for original_index, token in enumerate(tokens):

        if original_index == index:
            break

        word = token.lower()

        if (
            word.endswith("if")
            and word != "if"
            and len(word) > 2
        ):
            logical_index += 2
        else:
            logical_index += 1

    conditional_indexes = [
        i
        for i, token in enumerate(logical_tokens)
        if token.lower() == "if"
    ]

    hypothetical_indexes = [
        i
        for i, token in enumerate(logical_tokens)
        if token.lower() in HYPOTHETICAL_MARKERS
    ]

    if not conditional_indexes:
        return False

    if not hypothetical_indexes:
        return False

    for conditional_index in conditional_indexes:

        # -----------------------------------------------------
        # If ... would/were ... token
        #
        # Only treat tokens INSIDE the condition as conditional.
        # The result clause after the comma should remain scored.
        # -----------------------------------------------------

        if conditional_index < logical_index:

            has_hypothetical = any(
                conditional_index
                < hypothetical_index
                <= logical_index
                for hypothetical_index
                in hypothetical_indexes
            )

            if has_hypothetical:
                return True

        # -----------------------------------------------------
        # token ... if ... were ...
        #
        # Do NOT suppress the result before 'if'.
        #
        # Example:
        #
        # I would love this if the screen were better.
        #
        # 'love' must remain +2.
        # -----------------------------------------------------

        if conditional_index > logical_index:
            continue

    return False


def find_counterfactual_condition(
    tokens: list[str],
) -> dict[str, object]:
    """
    Detect counterfactual conditions that imply a current
    negative state.

    Examples:

        If the battery were better, I would love this.

        If the camera were amazing, I would like this.

        I would love this if the screen were better.

        If this were cheaper, it would be great.
    """

    logical_tokens = _split_attached_if(tokens)

    normalized = [
        token.lower()
        for token in logical_tokens
    ]

    if_indexes = [
        index
        for index, token in enumerate(normalized)
        if token == "if"
    ]

    if not if_indexes:
        return {
            "has_counterfactual": False,
            "condition_indexes": [],
            "improvement_indexes": [],
            "result_indexes": [],
            "negative_score": 0.0,
        }

    condition_indexes: list[int] = []
    improvement_indexes: list[int] = []
    result_indexes: list[int] = []

    for if_index in if_indexes:

        # -----------------------------------------------------
        # Find comma after "if".
        # -----------------------------------------------------

        comma_after = None

        for index in range(
            if_index + 1,
            len(logical_tokens),
        ):
            if logical_tokens[index] == ",":
                comma_after = index
                break

        # -----------------------------------------------------
        # Pattern:
        #
        # If condition, result
        # -----------------------------------------------------

        if comma_after is not None:

            condition_start = if_index + 1
            condition_end = comma_after

            result_start = comma_after + 1
            result_end = len(logical_tokens)

        # -----------------------------------------------------
        # Pattern:
        #
        # result if condition
        # -----------------------------------------------------

        else:

            condition_start = if_index + 1
            condition_end = len(logical_tokens)

            result_start = 0
            result_end = if_index

        # -----------------------------------------------------
        # Find improvement language.
        # -----------------------------------------------------

        local_improvements = [
            index
            for index in range(
                condition_start,
                condition_end,
            )
            if normalized[index]
            in IMPROVEMENT_WORDS
        ]

        if not local_improvements:
            continue

        # -----------------------------------------------------
        # Find positive result language.
        # -----------------------------------------------------

        local_results = [
            index
            for index in range(
                result_start,
                result_end,
            )
            if normalized[index]
            in RESULT_WORDS
        ]

        # Support:
        #
        # If this were cheaper, it would be great.
        #

        if not local_results:

            local_results = [
                index
                for index in range(
                    result_start,
                    result_end,
                )
                if normalized[index]
                in {
                    "good",
                    "great",
                    "amazing",
                    "excellent",
                    "wonderful",
                    "perfect",
                }
            ]

        if not local_results:
            continue

        condition_indexes.extend(
            range(
                condition_start,
                condition_end,
            )
        )

        improvement_indexes.extend(
            local_improvements
        )

        result_indexes.extend(
            local_results
        )

    condition_indexes = sorted(
        set(condition_indexes)
    )

    improvement_indexes = sorted(
        set(improvement_indexes)
    )

    result_indexes = sorted(
        set(result_indexes)
    )

    if not improvement_indexes or not result_indexes:
        return {
            "has_counterfactual": False,
            "condition_indexes": [],
            "improvement_indexes": [],
            "result_indexes": [],
            "negative_score": 0.0,
        }

    return {
        "has_counterfactual": True,
        "condition_indexes": condition_indexes,
        "improvement_indexes": improvement_indexes,
        "result_indexes": result_indexes,

        # Strong enough to override the positive hypothetical
        # result when the condition implies a current deficiency.
        "negative_score": -3.0,
    }