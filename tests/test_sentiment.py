from sentix import SentimentAnalyzer


def test_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this!"
    )

    assert result["label"] == "positive"


def test_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I hate this!"
    )

    assert result["label"] == "negative"


def test_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't love this."
    )

    assert result["label"] == "negative"


def test_intensifier():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict(
        "This is good."
    )

    intense = analyzer.predict(
        "This is extremely good."
    )

    assert intense["score"] > normal["score"]



def test_negation_edge_cases():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I don't hate this.")

    assert result["label"] == "positive"


def test_intensity_edge_cases():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("This is good.")

    intense = analyzer.predict("This is extremely good.")

    assert intense["score"] > normal["score"]
def test_negation_edge_cases():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I don't hate this.")

    assert result["label"] == "positive"


def test_capitalization():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("I love this.")

    shouting = analyzer.predict("I LOVE this.")

    assert shouting["score"] > normal["score"]

def test_repeated_characters():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("I love this.")

    stretched = analyzer.predict("I loooove this.")

    assert stretched["label"] == "positive"

from sentix.rules.phrases import find_phrases


def test_phrase_detection():
    tokens = ["This", "is", "a", "waste", "of", "money"]

    matches = find_phrases(tokens)

    assert len(matches) == 1
    assert matches[0][2] == -2.5    

def test_positive_phrase_detection():
    tokens = ["This", "is", "worth", "the", "money"]

    matches = find_phrases(tokens)

    assert len(matches) == 1
    assert matches[0][2] == 2.0

def test_not_bad():
    analyzer = SentimentAnalyzer()
    result = analyzer.predict("This is not bad.")

    assert result["label"] == "positive"


def test_not_good():
    analyzer = SentimentAnalyzer()
    result = analyzer.predict("This is not good.")

    assert result["label"] == "negative"


def test_negation_after_punctuation():
    analyzer = SentimentAnalyzer()
    result = analyzer.predict("This is good. Not bad.")

    assert result["label"] == "positive"

def test_negation_with_intensifier():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't really like this."
    )

    assert result["label"] == "negative"

def test_negation_scope():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is bad."
    )

    assert result["label"] == "positive"


def test_positive_sentence_without_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I think this is bad."
    )

    assert result["label"] == "negative"

def test_negation_stops_at_sentence_boundary():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like this. The camera is amazing."
    )

    assert result["label"] == "positive"

def test_negation_with_but():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like this, but the camera is amazing."
    )

    assert result["label"] == "positive"

def test_mixed_sentiment():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The design is amazing, but the battery is terrible."
    )

    assert result["label"] == "mixed"

def test_mostly_positive_not_mixed():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The design is amazing, but the battery is slightly bad."
    )

    assert result["label"] == "positive"

def test_mostly_negative_not_mixed():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The design is slightly bad, but the battery is terrible."
    )

    assert result["label"] == "negative"

def test_result_contains_sentiment_scores():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert "label" in result
    assert "score" in result
    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result

def test_result_scores_are_normalized():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    total = (
        result["positive"]
        + result["negative"]
        + result["neutral"]
    )

    assert abs(total - 1.0) < 0.001  

def test_neutral_sentence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I went to the store yesterday."
    )

    assert result["label"] == "neutral"

def test_neutral_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I went to the store yesterday."
    )

    assert result["neutral"] > 0.5
def test_weak_positive_is_not_strong_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is okay."
    )

    assert result["positive"] < 1.0              

def test_emotion_lexicon():
    from sentix.lexicon.emotion import EMOTION_LEXICON

    assert EMOTION_LEXICON["happy"]["joy"] == 1.0
    assert EMOTION_LEXICON["angry"]["anger"] == 1.0
    assert EMOTION_LEXICON["scared"]["fear"] == 1.0

def test_emotion_detection():
    from sentix.rules.emotion import detect_emotions

    tokens = ["I", "am", "happy"]

    emotions = detect_emotions(tokens)

    assert emotions["joy"] > 0    

def test_multiple_emotions():
    from sentix.rules.emotion import detect_emotions

    tokens = ["I", "am", "angry", "and", "scared"]

    emotions = detect_emotions(tokens)

    assert emotions["anger"] > 0
    assert emotions["fear"] > 0    

def test_sadness_emotion():
    from sentix.rules.emotion import detect_emotions

    tokens = ["I", "feel", "sad"]

    emotions = detect_emotions(tokens)

    assert emotions["sadness"] > 0    

def test_surprise_emotion():
    from sentix.rules.emotion import detect_emotions

    tokens = ["I", "am", "shocked"]

    emotions = detect_emotions(tokens)

    assert emotions["surprise"] > 0 

def test_predict_returns_emotions():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am happy!")

    assert "emotions" in result
    assert result["emotions"]["joy"] > 0


def test_predict_detects_multiple_emotions():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am angry and scared.")

    assert result["emotions"]["anger"] > 0
    assert result["emotions"]["fear"] > 0


def test_predict_no_emotions():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I bought a laptop yesterday.")

    assert result["emotions"]["joy"] == 0
    assert result["emotions"]["anger"] == 0
    assert result["emotions"]["sadness"] == 0  

def test_aspect_sentiment():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is excellent but the battery is terrible."
    )

    assert result["aspects"]["camera"]["sentiment"] == "positive"
    assert result["aspects"]["battery"]["sentiment"] == "negative"


def test_multiple_aspects():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The screen is amazing and the keyboard is great."
    )

    assert result["aspects"]["screen"]["sentiment"] == "positive"
    assert result["aspects"]["keyboard"]["sentiment"] == "positive"


def test_aspect_with_negative_sentiment():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is bad but the design is beautiful."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"
    assert result["aspects"]["design"]["sentiment"] == "positive"        

def test_aspect_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is not good."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"


def test_aspect_negated_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The battery is not terrible."
    )

    assert result["aspects"]["battery"]["sentiment"] == "positive"


def test_aspect_negation_does_not_cross_sentence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is not good. The battery is excellent."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"
    assert result["aspects"]["battery"]["sentiment"] == "positive" 

def test_aspect_intensifier():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is extremely good."
    )

    assert result["aspects"]["camera"]["sentiment"] == "positive"


def test_aspect_diminisher():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The battery is slightly bad."
    )

    assert result["aspects"]["battery"]["sentiment"] == "negative"


def test_aspect_intensity_multiple():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The screen is incredibly amazing but the keyboard is barely good."
    )

    assert result["aspects"]["screen"]["sentiment"] == "positive"
    assert result["aspects"]["keyboard"]["sentiment"] == "positive"

def test_aspect_negative_phrase():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is a waste of money."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"


def test_aspect_positive_phrase():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The battery is worth the money."
    )

    assert result["aspects"]["battery"]["sentiment"] == "positive"


def test_multiple_aspects_with_phrases():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is a waste of money but the screen is highly recommended."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"
    assert result["aspects"]["screen"]["sentiment"] == "positive"    

def test_aspect_returns_strength():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is extremely good."
    )

    assert result["aspects"]["camera"]["sentiment"] == "positive"
    assert result["aspects"]["camera"]["score"] > 1.0


def test_aspect_negative_strength():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The battery is slightly bad."
    )

    assert result["aspects"]["battery"]["sentiment"] == "negative"
    assert result["aspects"]["battery"]["score"] < 0


def test_aspect_negation_strength():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is not good."
    )

    assert result["aspects"]["camera"]["sentiment"] == "negative"
    assert result["aspects"]["camera"]["score"] < 0   

def test_confidence_exists():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert "confidence" in result


def test_confidence_is_between_zero_and_one():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I absolutely love this!")

    assert 0.0 <= result["confidence"] <= 1.0


def test_stronger_positive_has_higher_confidence():
    analyzer = SentimentAnalyzer()

    weak = analyzer.predict("good")
    strong = analyzer.predict("I absolutely love this!")

    assert strong["confidence"] > weak["confidence"]


def test_mixed_sentiment_has_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this but the battery is terrible."
    )

    assert result["label"] == "mixed"
    assert result["confidence"] > 0.0


def test_neutral_has_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I went to the store yesterday."
    )

    assert result["label"] == "neutral"
    assert result["confidence"] > 0.0

def test_confidence_stronger_intensity():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("good")
    intense = analyzer.predict("very good")
    extreme = analyzer.predict("extremely good")

    assert normal["label"] == "positive"
    assert intense["label"] == "positive"
    assert extreme["label"] == "positive"

    assert normal["confidence"] < intense["confidence"]
    assert intense["confidence"] < extreme["confidence"]


def test_confidence_stronger_negative_intensity():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("bad")
    intense = analyzer.predict("very bad")
    extreme = analyzer.predict("extremely bad")

    assert normal["label"] == "negative"
    assert intense["label"] == "negative"
    assert extreme["label"] == "negative"

    assert normal["confidence"] < intense["confidence"]
    assert intense["confidence"] < extreme["confidence"]


def test_confidence_conflicting_sentiment():
    analyzer = SentimentAnalyzer()

    positive = analyzer.predict("I love this.")
    conflicting = analyzer.predict(
        "I love this but the battery is terrible."
    )

    assert positive["label"] == "positive"
    assert conflicting["label"] == "mixed"

    assert conflicting["confidence"] > 0.0


def test_confidence_balanced_mixed_evidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The design is amazing but the battery is terrible."
    )

    assert result["label"] == "mixed"
    assert result["confidence"] > 0.8


def test_confidence_weak_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("good")

    assert result["label"] == "positive"
    assert 0.0 < result["confidence"] < 1.0


def test_confidence_weak_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("bad")

    assert result["label"] == "negative"
    assert 0.0 < result["confidence"] < 1.0


def test_confidence_neutral_without_evidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I went to the store yesterday."
    )

    assert result["label"] == "neutral"
    assert result["confidence"] == 0.5


def test_confidence_negated_sentiment():
    analyzer = SentimentAnalyzer()

    positive = analyzer.predict("I like this.")
    negated = analyzer.predict("I don't like this.")

    assert positive["label"] == "positive"
    assert negated["label"] == "negative"

    assert 0.0 <= positive["confidence"] <= 1.0
    assert 0.0 <= negated["confidence"] <= 1.0


def test_confidence_is_always_bounded():
    analyzer = SentimentAnalyzer()

    texts = [
        "good",
        "very good",
        "extremely amazing!",
        "bad",
        "extremely terrible!",
        "I don't like this.",
        "I love this but it is terrible.",
        "This is okay.",
        "I went to the store yesterday.",
    ]

    for text in texts:
        result = analyzer.predict(text)

        assert 0.0 <= result["confidence"] <= 1.0 

def test_confidence_increases_with_intensifier():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("good")
    intense = analyzer.predict("very good")

    assert intense["confidence"] > normal["confidence"]


def test_confidence_increases_with_stronger_intensifier():
    analyzer = SentimentAnalyzer()

    very = analyzer.predict("very good")
    extremely = analyzer.predict("extremely good")

    assert extremely["confidence"] > very["confidence"]


def test_confidence_increases_with_punctuation():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict("I love this.")
    excited = analyzer.predict("I love this!!!")

    assert excited["confidence"] >= normal["confidence"]


def test_strong_negative_evidence_has_high_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This product is absolutely terrible!"
    )

    assert result["label"] == "negative"
    assert result["confidence"] > 0.9


def test_multiple_positive_cues_have_high_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This product is amazing and fantastic!"
    )

    assert result["label"] == "positive"
    assert result["confidence"] > 0.9  

def test_uncertainty_reduces_confidence():
    analyzer = SentimentAnalyzer()

    direct = analyzer.predict(
        "This is good."
    )

    uncertain = analyzer.predict(
        "This might be good."
    )

    assert direct["label"] == "positive"
    assert uncertain["label"] == "positive"

    assert uncertain["confidence"] < direct["confidence"]


def test_strong_uncertainty_reduces_confidence():
    analyzer = SentimentAnalyzer()

    direct = analyzer.predict(
        "This is good."
    )

    uncertain = analyzer.predict(
        "This is probably good."
    )

    assert uncertain["confidence"] < direct["confidence"]


def test_certainty_increases_confidence():
    analyzer = SentimentAnalyzer()

    normal = analyzer.predict(
        "This is good."
    )

    certain = analyzer.predict(
        "This is definitely good."
    )

    assert certain["label"] == "positive"
    assert certain["confidence"] > normal["confidence"]


def test_uncertainty_does_not_change_basic_polarity():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This might be good."
    )

    assert result["label"] == "positive"


def test_uncertainty_is_not_negative_sentiment():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This might be good."
    )

    assert result["label"] != "negative"    
def test_negation_scope_with_think():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is good."
    )

    assert result["label"] == "negative"


def test_negation_scope_with_bad():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is bad."
    )

    assert result["label"] == "positive"


def test_negation_does_not_affect_second_clause():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like the camera, but the battery is excellent."
    )

    assert result["label"] == "positive"


def test_negation_with_multiple_sentiments():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like the camera, but I love the screen."
    )

    assert result["label"] == "positive"


def test_negation_does_not_cross_sentence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like the camera. I love the screen."
    )

    assert result["label"] == "mixed"


def test_positive_and_negative_clauses():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love the camera, but I hate the battery."
    )

    assert result["label"] == "mixed"


def test_double_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "It is not bad."
    )

    assert result["label"] == "positive"


def test_negated_negative_word():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "It is not terrible."
    )

    assert result["label"] == "positive"

def test_negation_with_intensity():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not very good."
    )

    assert result["label"] == "negative"


def test_negated_intense_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not extremely bad."
    )

    assert result["label"] == "positive"


def test_diminisher():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is slightly bad."
    )

    assert result["label"] == "negative"


def test_intensifier():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is extremely good."
    )

    assert result["label"] == "positive"


def test_negation_with_really():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I really don't like this."
    )

    assert result["label"] == "negative"


def test_negation_with_really_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't really like this."
    )

    assert result["label"] == "negative"


def test_triple_exclamation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is amazing!!!"
    )

    assert result["label"] == "positive"


def test_uppercase_sentiment():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is AMAZING."
    )

    assert result["label"] == "positive"


def test_negated_negative_with_intensity():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not very bad."
    )

    assert result["label"] == "positive"


def test_complex_mixed_aspects():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love the screen, but I hate the battery."
    )

    assert result["label"] == "mixed"    

def test_not_really_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't really like this."
    )

    assert result["label"] == "negative"


def test_really_not_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I really don't like this."
    )

    assert result["label"] == "negative"


def test_not_very_good():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not very good."
    )

    assert result["label"] == "negative"


def test_not_very_bad():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not very bad."
    )

    assert result["label"] == "positive"


def test_not_extremely_bad():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not extremely bad."
    )

    assert result["label"] == "positive"


def test_not_extremely_good():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "This is not extremely good."
    )

    assert result["label"] == "negative"


def test_never_liked():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I never liked this."
    )

    assert result["label"] == "negative"


def test_never_hated():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I never hated this."
    )

    assert result["label"] == "positive"


def test_sentence_boundary_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like this. This is amazing."
    )

    assert result["label"] == "positive"


def test_multiple_sentences_with_mixed_targets():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is terrible. The screen is amazing."
    )

    assert result["label"] == "mixed"

def test_liked_matches_like():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I liked this product.")

    assert result["label"] == "positive"


def test_loving_matches_love():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am loving this product.")

    assert result["label"] == "positive"


def test_hating_matches_hate():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am hating this product.")

    assert result["label"] == "negative"


def test_amazing_matches_amaze():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("This product is amazing.")

    assert result["label"] == "positive"


def test_amazed_matches_amazing():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I was amazed by this.")

    assert result["label"] == "positive"


def test_disappointed_matches_disappointing():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am disappointed with this product.")

    assert result["label"] == "negative"


def test_liked_with_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I did not like this.")

    assert result["label"] == "negative"


def test_loved_with_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I did not love this.")

    assert result["label"] == "negative"


def test_hated_with_negation():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I did not hate this.")

    assert result["label"] == "positive"


def test_repeated_morphology():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I looooved this!")

    assert result["label"] == "positive"  

def test_but_positive_wins():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I hate this, but the design is amazing."
    )

    assert result["label"] == "positive"


def test_but_negative_wins():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this, but the battery is terrible."
    )

    assert result["label"] == "negative"


def test_but_balanced_is_mixed():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love the camera, but I hate the battery."
    )

    assert result["label"] == "mixed"


def test_but_strong_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is bad, but the screen is extremely amazing."
    )

    assert result["label"] == "positive"


def test_but_strong_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is extremely terrible, but the screen is good."
    )

    assert result["label"] == "negative"


def test_however_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is terrible. However, the screen is amazing."
    )

    assert result["label"] == "positive"


def test_however_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is amazing. However, the battery is terrible."
    )

    assert result["label"] == "negative"


def test_although_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "Although the camera is bad, the screen is amazing."
    )

    assert result["label"] == "positive"


def test_although_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "Although the camera is amazing, the battery is terrible."
    )

    assert result["label"] == "negative"


def test_but_with_diminisher():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is amazing, but the battery is slightly bad."
    )

    assert result["label"] == "positive"

def test_dislike_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "I dislike this."
    )["label"] == "negative"

    assert analyzer.predict(
        "I disliked this."
    )["label"] == "negative"

    assert analyzer.predict(
        "I am disliking this."
    )["label"] == "negative"


def test_love_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "She loves this."
    )["label"] == "positive"

    assert analyzer.predict(
        "She loved this."
    )["label"] == "positive"

    assert analyzer.predict(
        "She is loving this."
    )["label"] == "positive"


def test_hate_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "He hates this."
    )["label"] == "negative"

    assert analyzer.predict(
        "He hated this."
    )["label"] == "negative"

    assert analyzer.predict(
        "He is hating this."
    )["label"] == "negative"


def test_like_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "I like this."
    )["label"] == "positive"

    assert analyzer.predict(
        "I liked this."
    )["label"] == "positive"

    assert analyzer.predict(
        "I am liking this."
    )["label"] == "positive"


def test_amaze_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "I am amazed by this."
    )["label"] == "positive"


def test_disappointed_morphology():
    analyzer = SentimentAnalyzer()

    assert analyzer.predict(
        "I was disappointed."
    )["label"] == "negative"

# ============================================================
# RESULT API TESTS
# ============================================================

def test_result_attribute_api():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert result.label == "positive"
    assert isinstance(result.score, float)
    assert isinstance(result.positive, float)
    assert isinstance(result.negative, float)
    assert isinstance(result.neutral, float)
    assert isinstance(result.confidence, float)


def test_result_contains_emotions():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I am happy.")

    assert isinstance(result.emotions, dict)
    assert "joy" in result.emotions


def test_result_contains_aspects():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The camera is amazing."
    )

    assert isinstance(result.aspects, dict)
    assert "camera" in result.aspects


def test_result_dictionary_access():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert result["label"] == "positive"
    assert result["score"] == result.score
    assert result["confidence"] == result.confidence


def test_result_contains_operator():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert "label" in result
    assert "score" in result
    assert "positive" in result
    assert "negative" in result
    assert "neutral" in result
    assert "confidence" in result
    assert "emotions" in result
    assert "aspects" in result


def test_result_get_method():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    assert result.get("label") == "positive"
    assert result.get("confidence") == result.confidence
    assert result.get("unknown") is None
    assert result.get("unknown", "default") == "default"


def test_result_to_dict():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    data = result.to_dict()

    assert isinstance(data, dict)

    assert data["label"] == result.label
    assert data["score"] == result.score
    assert data["positive"] == result.positive
    assert data["negative"] == result.negative
    assert data["neutral"] == result.neutral
    assert data["confidence"] == result.confidence
    assert data["emotions"] == result.emotions
    assert data["aspects"] == result.aspects


def test_result_to_dict_is_independent():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict("I love this.")

    data = result.to_dict()

    data["label"] = "changed"

    assert result.label == "positive"   

# ============================================================
# EXPLAINABILITY TESTS
# ============================================================

def test_sentiment_evidence():
    from sentix.explanation import SentimentEvidence

    evidence = SentimentEvidence(
        text="love",
        score=2.0,
        source="lexicon",
    )

    assert evidence.text == "love"
    assert evidence.score == 2.0
    assert evidence.source == "lexicon"
    assert evidence.modifier == 1.0
    assert evidence.negated is False


def test_sentiment_evidence_to_dict():
    from sentix.explanation import SentimentEvidence

    evidence = SentimentEvidence(
        text="love",
        score=2.0,
        source="lexicon",
        modifier=1.7,
        negated=False,
    )

    data = evidence.to_dict()

    assert data["text"] == "love"
    assert data["score"] == 2.0
    assert data["source"] == "lexicon"
    assert data["modifier"] == 1.7
    assert data["negated"] is False 

def test_scorer_collects_lexicon_evidence():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I love this."

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    assert score > 0
    assert positive > 0
    assert negative == 0

    assert len(evidence) >= 1

    love_evidence = [
        item
        for item in evidence
        if item.text == "love"
    ]

    assert len(love_evidence) == 1

    assert love_evidence[0].source == "lexicon"
    assert love_evidence[0].score == 2.0
    assert love_evidence[0].negated is False


def test_scorer_collects_negation_evidence():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I do not love this."

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    love_evidence = [
        item
        for item in evidence
        if item.text == "love"
    ]

    assert len(love_evidence) == 1
    assert love_evidence[0].negated is True
    assert love_evidence[0].score < 0


def test_scorer_collects_intensity_evidence():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I absolutely love this."

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    love_evidence = [
        item
        for item in evidence
        if item.text == "love"
    ]

    assert len(love_evidence) == 1

    assert love_evidence[0].modifier == 1.7
    assert love_evidence[0].score == 2.0 * 1.7


def test_scorer_collects_emoji_evidence():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "Amazing! 🔥"

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    emoji_evidence = [
        item
        for item in evidence
        if item.source == "emoji"
    ]

    assert len(emoji_evidence) == 1
    assert emoji_evidence[0].text == "🔥"   

def test_result_contains_evidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this! 🔥"
    )

    assert isinstance(result.evidence, list)
    assert len(result.evidence) >= 2


def test_result_evidence_contains_love():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this!"
    )

    love = [
        item
        for item in result.evidence
        if item.text == "love"
    ]

    assert len(love) == 1
    assert love[0].source == "lexicon"
    assert love[0].modifier == 1.7
    assert love[0].score == 3.4


def test_result_evidence_dictionary_access():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this."
    )

    assert "evidence" in result
    assert result["evidence"] == result.evidence


def test_result_to_dict_contains_evidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this! 🔥"
    )

    data = result.to_dict()

    assert "evidence" in data
    assert isinstance(data["evidence"], list)

    assert any(
        item["text"] == "love"
        for item in data["evidence"]
    )

def test_punctuation_evidence():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I love this!"

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    punctuation = [
        item
        for item in evidence
        if item.source == "punctuation"
    ]

    assert len(punctuation) == 1

    assert punctuation[0].text == "!"
    assert punctuation[0].modifier == 1.1
    assert punctuation[0].score == 1.1


def test_multiple_exclamation_marks_are_capped():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I love this!!!!!"

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    punctuation = [
        item
        for item in evidence
        if item.source == "punctuation"
    ]

    assert len(punctuation) == 1

    assert punctuation[0].modifier == 1.3
    assert punctuation[0].score == 1.3


def test_no_punctuation_evidence_without_exclamation():
    from sentix.tokenizer import tokenize
    from sentix.scorer import score_tokens_with_evidence

    text = "I love this."

    tokens = tokenize(text)

    score, positive, negative, evidence = (
        score_tokens_with_evidence(
            tokens,
            text
        )
    )

    punctuation = [
        item
        for item in evidence
        if item.source == "punctuation"
    ]

    assert punctuation == []


# ============================================================
# EXPLANATION API TESTS
# ============================================================

def test_result_explain_text():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this! 🔥"
    )

    explanation = result.explain()

    assert isinstance(explanation, str)

    assert "Sentix Analysis" in explanation
    assert "POSITIVE" in explanation
    assert "Confidence" in explanation
    assert "love" in explanation
    assert "🔥" in explanation
    assert "punctuation" in explanation


def test_result_explain_dict():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this! 🔥"
    )

    explanation = result.explain("dict")

    assert isinstance(explanation, dict)

    assert explanation["sentiment"] == result.label
    assert explanation["score"] == result.score
    assert explanation["confidence"] == result.confidence

    assert isinstance(
        explanation["evidence"],
        list
    )


def test_result_explain_json():
    import json

    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this!"
    )

    explanation = result.explain("json")

    assert isinstance(explanation, str)

    data = json.loads(explanation)

    assert data["sentiment"] == "positive"
    assert data["score"] == result.score
    assert "evidence" in data


def test_result_explain_invalid_format():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this."
    )

    try:
        result.explain("xml")
        assert False
    except ValueError:
        pass         
          

# ============================================================
# DETAILED EVIDENCE TESTS
# ============================================================

def test_evidence_contains_base_score():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love this."
    )

    love = [
        item
        for item in result.evidence
        if item.text == "love"
    ][0]

    assert love.base_score == 2.0
    assert love.intensity_modifier == 1.0
    assert love.capitalization_modifier == 1.0
    assert love.negation_modifier == 1.0
    assert love.score == 2.0


def test_evidence_contains_intensity_breakdown():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this."
    )

    love = [
        item
        for item in result.evidence
        if item.text == "love"
    ][0]

    assert love.base_score == 2.0
    assert love.intensity_modifier == 1.7
    assert love.capitalization_modifier == 1.0
    assert love.negation_modifier == 1.0
    assert love.score == 3.4


def test_evidence_contains_negation_breakdown():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I do not love this."
    )

    love = [
        item
        for item in result.evidence
        if item.text == "love"
    ][0]

    assert love.base_score == 2.0
    assert love.intensity_modifier == 1.0
    assert love.negation_modifier == -1.0
    assert love.negated is True
    assert love.score == -2.0


def test_evidence_contains_capitalization_breakdown():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I LOVE this."
    )

    love = [
        item
        for item in result.evidence
        if item.text == "LOVE"
    ][0]

    assert love.base_score == 2.0
    assert love.intensity_modifier == 1.0
    assert love.capitalization_modifier == 1.2
    assert love.negation_modifier == 1.0
    assert love.score == 2.4


def test_evidence_to_dict_contains_breakdown():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely LOVE this!"
    )

    love = [
        item
        for item in result.evidence
        if item.text == "LOVE"
    ][0]

    data = love.to_dict()

    assert data["base_score"] == 2.0
    assert data["intensity_modifier"] == 1.7
    assert data["capitalization_modifier"] == 1.2
    assert data["negation_modifier"] == 1.0
    assert data["negated"] is False
    assert data["score"] == 4.08


def test_explain_shows_detailed_breakdown():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely love this!"
    )

    explanation = result.explain()

    assert "Base score" in explanation
    assert "Intensity modifier" in explanation
    assert "Negation" in explanation
    assert "Final contribution" in explanation

def test_raw_score_is_available():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely LOVE this!"
    )

    assert result.raw_score is not None
    assert result.punctuation_modifier == 1.1


def test_raw_score_and_final_score_are_traceable():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely LOVE this!"
    )

    expected = (
        result.raw_score
        * result.punctuation_modifier
    )

    assert abs(
        expected - result.score
    ) < 1e-10


def test_to_dict_contains_score_trace():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely LOVE this!"
    )

    data = result.to_dict()

    assert data["raw_score"] == result.raw_score
    assert (
        data["punctuation_modifier"]
        == result.punctuation_modifier
    )


def test_explain_contains_score_trace():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I absolutely LOVE this!"
    )

    explanation = result.explain()

    assert "Raw score" in explanation
    assert "Punctuation" in explanation
    assert "Final score" in explanation

# ============================================================
# CONFIDENCE CALIBRATION TESTS
# ============================================================

def test_confidence_is_between_zero_and_one():
    analyzer = SentimentAnalyzer()

    examples = [
        "I love this.",
        "I hate this.",
        "This is okay.",
        "I absolutely love this!",
        "I don't like this.",
        "I love the camera, but I hate the battery.",
        "This is a completely ordinary sentence.",
    ]

    for text in examples:
        result = analyzer.predict(text)

        assert 0.0 <= result.confidence <= 1.0


def test_strong_positive_has_higher_confidence_than_weak_positive():
    analyzer = SentimentAnalyzer()

    weak = analyzer.predict(
        "I like this."
    )

    strong = analyzer.predict(
        "I absolutely love this!"
    )

    assert strong.confidence > weak.confidence


def test_strong_negative_has_higher_confidence_than_weak_negative():
    analyzer = SentimentAnalyzer()

    weak = analyzer.predict(
        "This is bad."
    )

    strong = analyzer.predict(
        "This is absolutely terrible!"
    )

    assert strong.confidence > weak.confidence


def test_neutral_without_evidence_has_neutral_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "The table is made of wood."
    )

    assert result.label == "neutral"
    assert result.confidence == 0.5


def test_uncertain_language_reduces_confidence():
    analyzer = SentimentAnalyzer()

    direct = analyzer.predict(
        "This is amazing."
    )

    uncertain = analyzer.predict(
        "Maybe this is amazing."
    )

    assert uncertain.confidence < direct.confidence


def test_certain_language_increases_confidence():
    analyzer = SentimentAnalyzer()

    direct = analyzer.predict(
        "This is amazing."
    )

    certain = analyzer.predict(
        "This is definitely amazing."
    )

    assert certain.confidence > direct.confidence


def test_balanced_mixed_sentiment_has_reasonable_confidence():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I love the camera, but I hate the battery."
    )

    assert result.label == "mixed"
    assert 0.0 <= result.confidence <= 1.0             

# ============================================================
# NEGATION SCOPE TESTS
# ============================================================

def test_double_negation_becomes_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is not good."
    )

    assert result.label == "positive"


def test_nested_negation_with_like():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I can't say that I don't like this."
    )

    assert result.label == "positive"


def test_simple_negation_remains_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't like this."
    )

    assert result.label == "negative"


def test_simple_negative_negation_becomes_positive():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't hate this."
    )

    assert result.label == "positive"


def test_existing_not_amazing_remains_negative():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is amazing."
    )

    assert result.label == "negative"


def test_negation_count_is_recorded():
    analyzer = SentimentAnalyzer()

    result = analyzer.predict(
        "I don't think this is not good."
    )

    good = [
        item
        for item in result.evidence
        if item.text.lower() == "good"
    ]

    assert len(good) == 1
    assert good[0].negation_count == 2
    assert good[0].negation_modifier == 1.0