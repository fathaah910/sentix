"""
Sentix Contrast Engine

Handles contrast markers:

    but
    however
    although
    though

The engine considers:

    - sentiment strength
    - clause type
    - explicit aspects
    - personal opinions
    - generic references
    - punctuation
    - temporal transitions
    - attached contrast markers
"""

from ..normalize import normalize_word
from ..lexicon.sentiment import SENTIMENT_LEXICON
from .intensity import get_intensity_modifier


# ============================================================
# CONTRAST MARKERS
# ============================================================

CONTRAST_MARKERS = {
    "but",
    "however",
    "although",
    "though",
}


# ============================================================
# TEMPORAL MARKERS
# ============================================================

TEMPORAL_LATER_WORDS = {
    "now",
    "today",
    "currently",
    "presently",
}

TEMPORAL_EARLIER_WORDS = {
    "first",
    "initially",
    "earlier",
    "yesterday",
    "before",
    "previously",
}


# ============================================================
# COMMON ASPECTS
# ============================================================

COMMON_ASPECTS = {
    "camera",
    "screen",
    "battery",
    "design",
    "keyboard",
    "display",
    "audio",
    "sound",
    "performance",
    "price",
    "quality",
    "phone",
    "product",
    "app",
}


# ============================================================
# PERSONAL OPINION WORDS
# ============================================================

OPINION_WORDS = {
    "love",
    "like",
    "hate",
    "dislike",
}


# ============================================================
# GENERIC REFERENCES
# ============================================================

GENERIC_REFERENCES = {
    "this",
    "that",
    "it",
    "thing",
    "stuff",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_token(token: str) -> str:
    return token.lower().strip(
        ".,!?;:-"
    )


def _token_score(token: str) -> float:
    normalized = normalize_word(
        token
    )

    return SENTIMENT_LEXICON.get(
        normalized,
        0.0,
    )


# ============================================================
# SIDE SCORE
# ============================================================

def _side_score(
    tokens: list[str],
) -> float:
    """
    Calculate sentiment score for one
    side of a contrast.
    """

    score = 0.0

    for index, token in enumerate(tokens):

        base_score = _token_score(
            token
        )

        if base_score == 0.0:
            continue

        intensity = get_intensity_modifier(
            tokens,
            index,
        )

        score += (
            base_score
            * intensity
        )

    return score


# ============================================================
# POLARITY
# ============================================================

def _polarity(
    score: float,
) -> str | None:

    if score > 0:
        return "positive"

    if score < 0:
        return "negative"

    return None


# ============================================================
# CLAUSE CLASSIFICATION
# ============================================================

def _classify_clause(
    tokens: list[str],
) -> str:
    """
    Classify clause as:

        opinion
        aspect
        generic
        other
    """

    normalized = {
        normalize_word(token)
        for token in tokens
    }

    if normalized & OPINION_WORDS:
        return "opinion"

    if normalized & COMMON_ASPECTS:
        return "aspect"

    if normalized & GENERIC_REFERENCES:
        return "generic"

    return "other"


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def _has_earlier_marker(
    tokens: list[str],
) -> bool:

    normalized = {
        _clean_token(token)
        for token in tokens
    }

    return bool(
        normalized
        & TEMPORAL_EARLIER_WORDS
    )


def _has_later_marker(
    tokens: list[str],
) -> bool:

    normalized = {
        _clean_token(token)
        for token in tokens
    }

    return bool(
        normalized
        & TEMPORAL_LATER_WORDS
    )


# ============================================================
# FIND CONTRAST MARKER
# ============================================================

def _find_contrast_marker(
    tokens: list[str],
) -> tuple[int | None, str | None]:

    for index, token in enumerate(tokens):

        word = token.lower()

        if word in CONTRAST_MARKERS:
            return index, word

        # Attached "but"
        if (
            word.startswith("but")
            and len(word) > 3
        ):
            return index, "but"

    return None, None


# ============================================================
# SPLIT ATTACHED MARKER
# ============================================================

def _split_attached_marker(
    tokens: list[str],
    marker_index: int,
    marker: str,
) -> tuple[list[str], list[str]]:

    token = tokens[
        marker_index
    ]

    word = token.lower()

    # --------------------------------------------------------
    # Attached BUT
    # --------------------------------------------------------

    if (
        marker == "but"
        and word.startswith("but")
        and word != "but"
    ):

        remainder = token[3:]

        before_tokens = tokens[
            :marker_index
        ]

        after_tokens = []

        if remainder:
            after_tokens.append(
                remainder
            )

        after_tokens.extend(
            tokens[
                marker_index + 1:
            ]
        )

        return (
            before_tokens,
            after_tokens,
        )

    # --------------------------------------------------------
    # Normal marker
    # --------------------------------------------------------

    return (
        tokens[:marker_index],
        tokens[marker_index + 1:],
    )


# ============================================================
# INITIAL ALTHOUGH / THOUGH
# ============================================================

def _split_initial_subordinate_clause(
    tokens: list[str],
) -> tuple[list[str], list[str]] | None:

    if not tokens:
        return None

    marker = tokens[0].lower()

    if marker not in {
        "although",
        "though",
    }:
        return None

    for index in range(
        1,
        len(tokens),
    ):

        if tokens[index] == ",":

            return (
                tokens[1:index],
                tokens[index + 1:],
            )

    return None


# ============================================================
# BUT DECISION ENGINE
# ============================================================

def _choose_but_label(
    before_tokens: list[str],
    after_tokens: list[str],
    has_comma_before: bool = False,
) -> str | None:

    before_score = _side_score(
        before_tokens
    )

    after_score = _side_score(
        after_tokens
    )

    # ========================================================
    # NO SENTIMENT
    # ========================================================

    if (
        before_score == 0
        and after_score == 0
    ):
        return None

    # ========================================================
    # ONLY BEFORE
    # ========================================================

    if (
        before_score != 0
        and after_score == 0
    ):
        return _polarity(
            before_score
        )

    # ========================================================
    # ONLY AFTER
    # ========================================================

    if (
        before_score == 0
        and after_score != 0
    ):
        return _polarity(
            after_score
        )

    # ========================================================
    # SAME POLARITY
    # ========================================================

    if (
        before_score > 0
        and after_score > 0
    ):
        return "positive"

    if (
        before_score < 0
        and after_score < 0
    ):
        return "negative"

    # ========================================================
    # STRENGTH
    # ========================================================

    before_strength = abs(
        before_score
    )

    after_strength = abs(
        after_score
    )

    weaker = min(
        before_strength,
        after_strength,
    )

    stronger = max(
        before_strength,
        after_strength,
    )

    before_type = _classify_clause(
        before_tokens
    )

    after_type = _classify_clause(
        after_tokens
    )

    # ========================================================
    # EQUAL STRENGTH
    # ========================================================

    if before_strength == after_strength:

        # ----------------------------------------------------
        # OPINION + OPINION
        #
        # I love the camera,
        # but I hate the battery.
        #
        # -> mixed
        # ----------------------------------------------------

        if (
            before_type == "opinion"
            and after_type == "opinion"
        ):
            return "mixed"

        # ----------------------------------------------------
        # OPINION + ASPECT
        #
        # I hate this,
        # but the design is amazing.
        #
        # -> positive
        #
        # I love this,
        # but the battery is terrible.
        #
        # -> negative
        #
        # The comma explicitly separates the
        # opinion from the concrete aspect.
        # ----------------------------------------------------

        if (
            before_type == "opinion"
            and after_type == "aspect"
        ):

            if has_comma_before:
                return _polarity(
                    after_score
                )

            return "mixed"

        # ----------------------------------------------------
        # ASPECT + OPINION
        #
        # The camera is terrible,
        # but I love it.
        #
        # -> positive
        # ----------------------------------------------------

        if (
            before_type == "aspect"
            and after_type == "opinion"
        ):

            if has_comma_before:
                return _polarity(
                    after_score
                )

            return "mixed"

        # ----------------------------------------------------
        # GENERIC + ASPECT
        #
        # I love this but the battery
        # is terrible.
        #
        # -> mixed
        #
        # I love this, but the battery
        # is terrible.
        #
        # -> negative
        # ----------------------------------------------------

        if (
            before_type == "generic"
            and after_type == "aspect"
        ):

            if has_comma_before:
                return _polarity(
                    after_score
                )

            return "mixed"

        # ----------------------------------------------------
        # GENERIC + OPINION
        # ----------------------------------------------------

        if (
            before_type == "generic"
            and after_type == "opinion"
        ):

            if has_comma_before:
                return _polarity(
                    after_score
                )

            return "mixed"

        # ----------------------------------------------------
        # OPINION + GENERIC
        # ----------------------------------------------------

        if (
            before_type == "opinion"
            and after_type == "generic"
        ):
            return "mixed"

        # ----------------------------------------------------
        # ASPECT + ASPECT
        #
        # Equal opposing aspects -> mixed.
        # ----------------------------------------------------

        if (
            before_type == "aspect"
            and after_type == "aspect"
        ):
            return "mixed"

        # ----------------------------------------------------
        # OTHER + ASPECT
        # ----------------------------------------------------

        if (
            before_type == "other"
            and after_type == "aspect"
        ):

            if has_comma_before:
                return _polarity(
                    after_score
                )

            return "mixed"

        # ----------------------------------------------------
        # ASPECT + OTHER
        # ----------------------------------------------------

        if (
            before_type == "aspect"
            and after_type == "other"
        ):

            if has_comma_before:
                return _polarity(
                    before_score
                )

            return "mixed"

        # ----------------------------------------------------
        # Default balanced conflict.
        # ----------------------------------------------------

        return "mixed"

    # ========================================================
    # OPINION + OPINION
    # ========================================================

    if (
        before_type == "opinion"
        and after_type == "opinion"
    ):
        return "mixed"

    # ========================================================
    # GENERIC + ASPECT
    # ========================================================

    if (
        before_type == "generic"
        and after_type == "aspect"
    ):

        if has_comma_before:
            return _polarity(
                after_score
            )

        if (
            stronger >= 1.5 * weaker
        ):
            return _polarity(
                after_score
            )

        return "mixed"

    # ========================================================
    # ASPECT + ASPECT
    # ========================================================

    if (
        before_type == "aspect"
        and after_type == "aspect"
    ):

        if (
            weaker > 0
            and stronger >= 1.5 * weaker
        ):

            if (
                before_strength
                > after_strength
            ):
                return _polarity(
                    before_score
                )

            return _polarity(
                after_score
            )

        return "mixed"

    # ========================================================
    # OPINION + ASPECT
    # ========================================================

    if (
        before_type == "opinion"
        and after_type == "aspect"
    ):
        return _polarity(
            after_score
        )

    # ========================================================
    # ASPECT + OPINION
    # ========================================================

    if (
        before_type == "aspect"
        and after_type == "opinion"
    ):
        return _polarity(
            after_score
        )

    # ========================================================
    # GENERIC + OPINION
    # ========================================================

    if (
        before_type == "generic"
        and after_type == "opinion"
    ):
        return _polarity(
            after_score
        )

    # ========================================================
    # OPINION + GENERIC
    # ========================================================

    if (
        before_type == "opinion"
        and after_type == "generic"
    ):
        return _polarity(
            after_score
        )

    # ========================================================
    # OTHER + ASPECT
    # ========================================================

    if (
        before_type == "other"
        and after_type == "aspect"
    ):

        if (
            stronger >= 1.5 * weaker
        ):
            return _polarity(
                after_score
            )

        return "mixed"

    # ========================================================
    # ASPECT + OTHER
    # ========================================================

    if (
        before_type == "aspect"
        and after_type == "other"
    ):

        if (
            stronger >= 1.5 * weaker
        ):
            return _polarity(
                before_score
            )

        return "mixed"

    # ========================================================
    # GENERAL STRENGTH DOMINANCE
    # ========================================================

    if (
        weaker > 0
        and stronger >= 1.5 * weaker
    ):

        if (
            before_strength
            > after_strength
        ):
            return _polarity(
                before_score
            )

        return _polarity(
            after_score
        )

    # ========================================================
    # BALANCED CONFLICT
    # ========================================================

    if (
        before_strength > 0
        and after_strength > 0
    ):
        return "mixed"

    # ========================================================
    # FALLBACK
    # ========================================================

    return _polarity(
        after_score
    )


# ============================================================
# ANALYZE BUT
# ============================================================

def _analyze_but(
    tokens: list[str],
    marker_index: int,
) -> dict:

    before_tokens, after_tokens = (
        _split_attached_marker(
            tokens,
            marker_index,
            "but",
        )
    )

    before_score = _side_score(
        before_tokens
    )

    after_score = _side_score(
        after_tokens
    )

    has_comma_before = (
        marker_index > 0
        and tokens[
            marker_index - 1
        ] == ","
    )

    label = _choose_but_label(
        before_tokens,
        after_tokens,
        has_comma_before=(
            has_comma_before
        ),
    )

    if (
        abs(after_score)
        >= abs(before_score)
    ):
        final_score = after_score
    else:
        final_score = before_score

    return {
        "has_contrast": True,
        "marker": "but",
        "marker_index": marker_index,
        "before_score": before_score,
        "after_score": after_score,
        "label": label,
        "score": final_score,
    }


# ============================================================
# ANALYZE HOWEVER / ALTHOUGH / THOUGH
# ============================================================

def _analyze_other_marker(
    tokens: list[str],
    marker_index: int,
    marker: str,
) -> dict:

    before_tokens = tokens[
        :marker_index
    ]

    after_tokens = tokens[
        marker_index + 1:
    ]

    before_score = _side_score(
        before_tokens
    )

    after_score = _side_score(
        after_tokens
    )

    # Later/main clause has priority.

    if after_score > 0:

        label = "positive"

    elif after_score < 0:

        label = "negative"

    elif before_score > 0:

        label = "positive"

    elif before_score < 0:

        label = "negative"

    else:

        label = "neutral"

    if (
        abs(after_score)
        >= abs(before_score)
    ):
        final_score = after_score
    else:
        final_score = before_score

    return {
        "has_contrast": True,
        "marker": marker,
        "marker_index": marker_index,
        "before_score": before_score,
        "after_score": after_score,
        "label": label,
        "score": final_score,
    }


# ============================================================
# MAIN CONTRAST ANALYZER
# ============================================================

def analyze_contrast(
    tokens: list[str],
) -> dict[str, object]:

    if not tokens:

        return {
            "has_contrast": False,
            "marker": None,
            "marker_index": None,
            "before_score": 0.0,
            "after_score": 0.0,
            "label": None,
            "score": 0.0,
        }

    marker_index, marker = (
        _find_contrast_marker(
            tokens
        )
    )

    # --------------------------------------------------------
    # No contrast marker.
    # --------------------------------------------------------

    if marker_index is None:

        return {
            "has_contrast": False,
            "marker": None,
            "marker_index": None,
            "before_score": 0.0,
            "after_score": 0.0,
            "label": None,
            "score": 0.0,
        }

    # ========================================================
    # INITIAL ALTHOUGH / THOUGH
    # ========================================================

    if (
        marker in {
            "although",
            "though",
        }
        and marker_index == 0
    ):

        split = (
            _split_initial_subordinate_clause(
                tokens
            )
        )

        if split is not None:

            before_tokens, after_tokens = (
                split
            )

            before_score = _side_score(
                before_tokens
            )

            after_score = _side_score(
                after_tokens
            )

            if after_score > 0:

                label = "positive"

            elif after_score < 0:

                label = "negative"

            elif before_score > 0:

                label = "positive"

            elif before_score < 0:

                label = "negative"

            else:

                label = "neutral"

            if (
                abs(after_score)
                >= abs(before_score)
            ):
                final_score = after_score
            else:
                final_score = before_score

            return {
                "has_contrast": True,
                "marker": marker,
                "marker_index": marker_index,
                "before_score": before_score,
                "after_score": after_score,
                "label": label,
                "score": final_score,
            }

    # ========================================================
    # BUT
    # ========================================================

    if marker == "but":

        return _analyze_but(
            tokens,
            marker_index,
        )

    # ========================================================
    # HOWEVER / ALTHOUGH / THOUGH
    # ========================================================

    return _analyze_other_marker(
        tokens,
        marker_index,
        marker,
    )