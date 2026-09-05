import math

from .certainty import get_certainty_modifier


def calculate_confidence(
    positive: float,
    negative: float,
    label: str,
    tokens: list[str] | None = None,
) -> float:
    """
    Calculate confidence in the predicted sentiment.

    Confidence is based on:

    1. Amount of sentiment evidence.
    2. Dominance of the predicted sentiment.
    3. Certainty or uncertainty expressed in the text.

    Returns a value between 0.0 and 1.0.
    """

    evidence = positive + negative

    # No sentiment evidence.
    if evidence == 0:
        return 0.5 if label == "neutral" else 0.0

    # Calculate polarity dominance.
    if label == "positive":
        dominance = positive / evidence

    elif label == "negative":
        dominance = negative / evidence

    elif label == "mixed":
        dominance = 1.0 - abs(positive - negative) / evidence

    else:
        dominance = 0.0

    # Convert evidence strength into a bounded value.
    evidence_strength = 1.0 - math.exp(-evidence)

    confidence = dominance * evidence_strength

    # Apply certainty / uncertainty information.
    if tokens is not None:
        certainty_modifier = get_certainty_modifier(tokens)
        confidence *= certainty_modifier

    return min(max(confidence, 0.0), 1.0)