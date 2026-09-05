from .tokenizer import tokenize

from .scorer import (
    score_tokens_with_evidence,
    punctuation_modifier,
)

from .confidence import calculate_confidence
from .result import SentimentResult

from .rules.emotion import detect_emotions
from .rules.aspect import detect_aspects
from .rules.context import analyze_context
from .rules.contrast import analyze_contrast


def normalize_scores(
    positive: float,
    negative: float,
) -> tuple[float, float, float]:
    """
    Convert positive / negative evidence into
    normalized probabilities.

    Neutral receives the remaining probability.
    """

    total = positive + negative

    if total == 0:
        return 0.0, 0.0, 1.0

    positive_ratio = positive / total
    negative_ratio = negative / total

    return (
        positive_ratio,
        negative_ratio,
        0.0,
    )


class SentimentAnalyzer:

    def predict(
        self,
        text: str,
    ) -> SentimentResult:

        # ====================================================
        # TOKENIZATION
        # ====================================================

        tokens = tokenize(text)

        # ====================================================
        # BASE SENTIMENT SCORING
        # ====================================================

        (
            score,
            positive,
            negative,
            evidence,
        ) = score_tokens_with_evidence(
            tokens,
            text,
        )

        # ====================================================
        # PUNCTUATION
        # ====================================================

        punctuation_factor = (
            punctuation_modifier(text)
        )

        if punctuation_factor != 1.0:
            raw_score = (
                score / punctuation_factor
            )
        else:
            raw_score = score

        # ====================================================
        # EMOTIONS
        # ====================================================

        emotions = detect_emotions(
            tokens
        )

        # ====================================================
        # CONTEXT
        # ====================================================

        context = analyze_context(
            tokens
        )

        # ====================================================
        # CONTRAST
        # ====================================================

        contrast = analyze_contrast(
            tokens
        )

        # ====================================================
        # NORMALIZED PROBABILITIES
        # ====================================================

        (
            positive_prob,
            negative_prob,
            neutral_prob,
        ) = normalize_scores(
            positive,
            negative,
        )

        # ====================================================
        # LABEL DECISION
        # ====================================================

        label = None

        # ----------------------------------------------------
        # CONTRAST HAS HIGH PRIORITY
        #
        # If the contrast engine has identified a semantic
        # relationship, use its decision before falling back
        # to the global sentiment score.
        # ----------------------------------------------------

        if (
            contrast.get("has_contrast")
            and contrast.get("label") is not None
        ):
            label = contrast["label"]

        # ----------------------------------------------------
        # DIFFERENT SENTENCE TARGETS
        #
        # Example:
        #
        # The camera is terrible.
        # The screen is amazing.
        #
        # -> mixed
        # ----------------------------------------------------

        elif (
            context.get(
                "different_sentence_targets",
                False,
            )
            and positive > 0.5
            and negative > 0.5
        ):
            label = "mixed"

        # ----------------------------------------------------
        # GENERAL MIXED SENTIMENT
        # ----------------------------------------------------

        elif (
            positive > 0.5
            and negative > 0.5
            and abs(
                positive - negative
            ) < 0.75
        ):
            label = "mixed"

        # ----------------------------------------------------
        # POSITIVE
        # ----------------------------------------------------

        elif score > 0.05:
            label = "positive"

        # ----------------------------------------------------
        # NEGATIVE
        # ----------------------------------------------------

        elif score < -0.05:
            label = "negative"

        # ----------------------------------------------------
        # NEUTRAL
        # ----------------------------------------------------

        else:
            label = "neutral"

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = calculate_confidence(
            positive,
            negative,
            label,
            tokens,
        )

        # ====================================================
        # RESULT
        # ====================================================

        return SentimentResult(
            label=label,
            score=score,
            positive=positive_prob,
            negative=negative_prob,
            neutral=neutral_prob,
            confidence=confidence,
            emotions=emotions,
            aspects=detect_aspects(tokens),
            evidence=evidence,
            raw_score=raw_score,
            punctuation_modifier=punctuation_factor,
        )