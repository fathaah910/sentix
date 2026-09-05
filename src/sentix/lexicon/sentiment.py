"""
Core sentiment lexicon for Sentix.

Positive values represent positive sentiment.
Negative values represent negative sentiment.

Approximate strength scale:

    0.5  -> very weak
    1.0  -> mild
    1.5  -> moderate
    2.0  -> strong
    2.5+ -> very strong
"""

SENTIMENT_LEXICON = {

    # =========================================================
    # POSITIVE SENTIMENT
    # =========================================================

    # Mild positive
    "good": 1.0,
    "nice": 1.0,

    # Moderate positive
    "great": 1.5,
    "happy": 1.5,
    "beautiful": 1.5,
    "like": 1.2,

    # Strong positive
    "excellent": 2.0,
    "amazing": 2.0,
    "awesome": 2.0,
    "fantastic": 2.0,
    "love": 2.0,
    "best": 2.0,

    # =========================================================
    # NEGATIVE SENTIMENT
    # =========================================================

    # Mild / moderate negative
    "bad": -1.0,
    "poor": -1.0,
    "dislike": -1.2,
    "sad": -1.5,
    "ugly": -1.5,
    "disappointing": -1.5,

    # Strong negative
    "terrible": -2.0,
    "awful": -2.0,
    "horrible": -2.0,
    "hate": -2.0,

    # Very strong negative
    "worst": -2.5,
}