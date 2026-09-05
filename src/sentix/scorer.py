from .lexicon.sentiment import SENTIMENT_LEXICON
from .lexicon.emoji import EMOJI_LEXICON

from .rules.negation import (
    get_negation_scope,
)

from .rules.intensity import (
    get_intensity_modifier,
)

from .rules.repetition import (
    normalize_repeated_characters,
)

from .rules.phrases import (
    find_phrases,
)

from .rules.conditional import (
    is_conditional_context,
    find_counterfactual_condition,
)

from .rules.contradiction import (
    detect_contradiction,
)

from .rules.reversal import (
    detect_reversal,
)

from .normalize import (
    normalize_word,
)

from .explanation import (
    SentimentEvidence,
)


def punctuation_modifier(
    text: str,
) -> float:
    """
    Calculate punctuation intensity.

    More exclamation marks increase sentiment strength,
    capped at 1.3x.
    """

    count = text.count("!")

    if count == 0:
        return 1.0

    return min(
        1.0 + (0.1 * count),
        1.3,
    )


def add_evidence(
    positive: float,
    negative: float,
    value: float,
) -> tuple[float, float]:
    """
    Add a signed sentiment value to positive/negative
    evidence buckets.
    """

    if value > 0:
        positive += value

    elif value < 0:
        negative += abs(value)

    return positive, negative


def score_tokens_with_evidence(
    tokens: list[str],
    text: str,
) -> tuple[
    float,
    float,
    float,
    list[SentimentEvidence],
]:
    """
    Score tokens while preserving detailed evidence.

    Returns:

        score
        positive evidence
        negative evidence
        evidence list
    """

    positive = 0.0
    negative = 0.0

    evidence: list[SentimentEvidence] = []

    # =========================================================
    # CONTEXT DETECTORS
    # =========================================================

    contradiction = detect_contradiction(tokens)

    contradiction_positive_indexes = set(
        contradiction["positive_indexes"]
    )

    reversal = detect_reversal(tokens)

    reversal_outcome_indexes = set(
        reversal["outcome_indexes"]
    )

    reversal_negative_indexes = set(
        reversal["negative_indexes"]
    )

    # Prevent unused-variable warnings while keeping the
    # contextual information available for future rules.
    _ = reversal_negative_indexes

    counterfactual = (
        find_counterfactual_condition(tokens)
    )

    # =========================================================
    # PHRASE SCORING
    # =========================================================

    phrase_matches = find_phrases(tokens)

    phrase_token_indices: set[int] = set()

    for start, end, phrase_score in phrase_matches:

        phrase_is_conditional = any(
            is_conditional_context(
                tokens,
                index,
            )
            for index in range(start, end)
        )

        if phrase_is_conditional:
            phrase_score = 0.0

        positive, negative = add_evidence(
            positive,
            negative,
            phrase_score,
        )

        phrase_token_indices.update(
            range(start, end)
        )

        evidence.append(
            SentimentEvidence(
                text=" ".join(
                    tokens[start:end]
                ),
                score=phrase_score,
                source="phrase",
                base_score=phrase_score,
                conditional=phrase_is_conditional,
            )
        )

    # =========================================================
    # TOKEN SCORING
    # =========================================================

    for index, token in enumerate(tokens):

        # -----------------------------------------------------
        # Skip tokens already consumed by a phrase.
        # -----------------------------------------------------

        if index in phrase_token_indices:
            continue

        # -----------------------------------------------------
        # Normalize repeated characters.
        #
        # Example:
        #
        # amaaaazing -> amazing
        # goooood     -> good
        # -----------------------------------------------------

        repeated_normalized = (
            normalize_repeated_characters(
                token
            )
        )

        normalized_token = normalize_word(
            repeated_normalized
        )

        # =====================================================
        # SENTIMENT LEXICON
        # =====================================================

        if normalized_token in SENTIMENT_LEXICON:

            base_score = (
                SENTIMENT_LEXICON[
                    normalized_token
                ]
            )

            # -------------------------------------------------
            # Intensity
            # -------------------------------------------------

            intensity_modifier = (
                get_intensity_modifier(
                    tokens,
                    index,
                )
            )

            word_score = (
                base_score
                * intensity_modifier
            )

            # -------------------------------------------------
            # Negation
            # -------------------------------------------------

            negation_scope = (
                get_negation_scope(
                    tokens,
                    index,
                )
            )

            negation_modifier = (
                negation_scope["modifier"]
            )

            negated = (
                negation_scope["negated"]
            )

            negation_count = (
                negation_scope["count"]
            )

            if negation_modifier == -1.0:
                word_score *= -1.0

            # -------------------------------------------------
            # Capitalization
            # -------------------------------------------------

            capitalization_modifier = 1.0

            if token.isupper() and len(token) > 1:

                capitalization_modifier = 1.2

                word_score *= (
                    capitalization_modifier
                )

            # -------------------------------------------------
            # Conditional context
            # -------------------------------------------------

            conditional = (
                is_conditional_context(
                    tokens,
                    index,
                )
            )

            if conditional:
                word_score = 0.0

            # -------------------------------------------------
            # Contradiction context
            #
            # Example:
            #
            # Wonderful, the app crashed.
            #
            # "Wonderful" should not dominate the
            # contextual negative signal.
            # -------------------------------------------------

            contradictory = (
                index
                in contradiction_positive_indexes
            )

            if (
                contradictory
                and word_score > 0
            ):
                word_score = 0.0

            # -------------------------------------------------
            # Add token evidence.
            # -------------------------------------------------

            positive, negative = add_evidence(
                positive,
                negative,
                word_score,
            )

            evidence.append(
                SentimentEvidence(
                    text=token,
                    score=word_score,
                    source="lexicon",
                    modifier=intensity_modifier,
                    base_score=base_score,
                    intensity_modifier=(
                        intensity_modifier
                    ),
                    capitalization_modifier=(
                        capitalization_modifier
                    ),
                    negation_modifier=(
                        negation_modifier
                    ),
                    negated=negated,
                    negation_count=(
                        negation_count
                    ),
                    conditional=conditional,
                )
            )

        # =====================================================
        # EMOJI
        # =====================================================

        elif token in EMOJI_LEXICON:

            emoji_score = (
                EMOJI_LEXICON[token]
            )

            positive, negative = add_evidence(
                positive,
                negative,
                emoji_score,
            )

            evidence.append(
                SentimentEvidence(
                    text=token,
                    score=emoji_score,
                    source="emoji",
                    base_score=emoji_score,
                )
            )

    # =========================================================
    # COUNTERFACTUAL EVIDENCE
    # =========================================================

    if counterfactual[
        "has_counterfactual"
    ]:

        counterfactual_score = (
            counterfactual[
                "negative_score"
            ]
        )

        positive, negative = add_evidence(
            positive,
            negative,
            counterfactual_score,
        )

        improvement_text = " ".join(
            tokens[index]
            for index in counterfactual[
                "improvement_indexes"
            ]
        )

        evidence.append(
            SentimentEvidence(
                text=improvement_text,
                score=counterfactual_score,
                source="conditional",
                base_score=counterfactual_score,
                conditional=True,
            )
        )

    # =========================================================
    # REVERSAL EVIDENCE
    # =========================================================

    if reversal["has_reversal"]:

        reversal_score = (
            reversal["positive_score"]
        )

        positive, negative = add_evidence(
            positive,
            negative,
            reversal_score,
        )

        outcome_text = " ".join(
            tokens[index]
            for index in sorted(
                reversal_outcome_indexes
            )
        )

        evidence.append(
            SentimentEvidence(
                text=outcome_text,
                score=reversal_score,
                source="reversal",
                base_score=reversal_score,
            )
        )

    # =========================================================
    # CONTRADICTION EVIDENCE
    # =========================================================

    if contradiction[
        "has_contradiction"
    ]:

        contextual_score = (
            contradiction[
                "negative_score"
            ]
        )

        positive, negative = add_evidence(
            positive,
            negative,
            contextual_score,
        )

        evidence.append(
            SentimentEvidence(
                text="contextual contradiction",
                score=contextual_score,
                source="context",
                base_score=contextual_score,
            )
        )

    # =========================================================
    # PUNCTUATION
    # =========================================================

    modifier = punctuation_modifier(
        text
    )

    positive *= modifier
    negative *= modifier

    if modifier != 1.0:

        evidence.append(
            SentimentEvidence(
                text="!",
                score=modifier,
                source="punctuation",
                modifier=modifier,
            )
        )

    # =========================================================
    # FINAL SCORE
    # =========================================================

    score = (
        positive
        - negative
    )

    return (
        score,
        positive,
        negative,
        evidence,
    )


def score_tokens(
    tokens: list[str],
    text: str,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Backward-compatible scoring API.

    Returns:

        score
        positive evidence
        negative evidence
    """

    (
        score,
        positive,
        negative,
        _,
    ) = score_tokens_with_evidence(
        tokens,
        text,
    )

    return (
        score,
        positive,
        negative,
    )