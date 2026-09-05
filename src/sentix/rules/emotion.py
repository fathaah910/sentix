from ..lexicon.emotion import EMOTION_LEXICON


def detect_emotions(tokens: list[str]) -> dict[str, float]:
    """
    Detect emotions from a list of tokens.

    Returns emotion scores for all supported emotions.
    """

    emotions = {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
    }

    for token in tokens:

        normalized_token = token.lower()

        if normalized_token in EMOTION_LEXICON:

            emotion_scores = EMOTION_LEXICON[normalized_token]

            for emotion, score in emotion_scores.items():
                emotions[emotion] += score

    return emotions