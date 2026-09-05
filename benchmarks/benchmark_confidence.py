from sentix import SentimentAnalyzer


analyzer = SentimentAnalyzer()


TEST_CASES = [

    # =========================================================
    # BASIC POSITIVE
    # =========================================================

    ("I love this.", "positive"),
    ("I like this.", "positive"),
    ("This is good.", "positive"),
    ("This is nice.", "positive"),
    ("This is great.", "positive"),
    ("This is amazing.", "positive"),
    ("This is excellent.", "positive"),
    ("This is awesome.", "positive"),
    ("This is fantastic.", "positive"),
    ("This is beautiful.", "positive"),

    # =========================================================
    # BASIC NEGATIVE
    # =========================================================

    ("I hate this.", "negative"),
    ("I dislike this.", "negative"),
    ("This is bad.", "negative"),
    ("This is poor.", "negative"),
    ("This is disappointing.", "negative"),
    ("This is sad.", "negative"),
    ("This is ugly.", "negative"),
    ("This is terrible.", "negative"),
    ("This is awful.", "negative"),
    ("This is horrible.", "negative"),

    # =========================================================
    # INTENSITY
    # =========================================================

    ("I really love this.", "positive"),
    ("I very much love this.", "positive"),
    ("I absolutely love this.", "positive"),
    ("I extremely love this.", "positive"),
    ("I incredibly love this.", "positive"),

    ("I really hate this.", "negative"),
    ("I absolutely hate this.", "negative"),
    ("This is extremely terrible.", "negative"),
    ("This is incredibly awful.", "negative"),
    ("This is very bad.", "negative"),

    # =========================================================
    # DIMINISHERS
    # =========================================================

    ("I slightly like this.", "positive"),
    ("I somewhat like this.", "positive"),
    ("I barely like this.", "positive"),
    ("I slightly dislike this.", "negative"),
    ("I somewhat dislike this.", "negative"),

    # =========================================================
    # NEGATION
    # =========================================================

    ("I do not love this.", "negative"),
    ("I don't love this.", "negative"),
    ("I do not like this.", "negative"),
    ("I don't like this.", "negative"),
    ("This is not good.", "negative"),
    ("This is not amazing.", "negative"),
    ("This is not bad.", "positive"),
    ("This is not terrible.", "positive"),
    ("I never loved this.", "negative"),
    ("I never liked this.", "negative"),

    # =========================================================
    # DOUBLE NEGATION / COMPLEX NEGATION
    # =========================================================

    ("It is not bad.", "positive"),
    ("It is not terrible.", "positive"),
    ("I don't think this is bad.", "positive"),
    ("I don't think this is good.", "negative"),
    ("I can't say this is bad.", "positive"),

    # =========================================================
    # UNCERTAINTY
    # =========================================================

    ("Maybe this is amazing.", "positive"),
    ("Perhaps this is amazing.", "positive"),
    ("Possibly this is good.", "positive"),
    ("This might be good.", "positive"),
    ("This could be amazing.", "positive"),
    ("This is probably good.", "positive"),
    ("This seems good.", "positive"),
    ("I think this is amazing.", "positive"),
    ("I believe this is good.", "positive"),
    ("I guess this is nice.", "positive"),

    # =========================================================
    # CERTAINTY
    # =========================================================

    ("This is definitely amazing.", "positive"),
    ("This is certainly amazing.", "positive"),
    ("This is clearly excellent.", "positive"),
    ("This is undoubtedly fantastic.", "positive"),
    ("This is surely good.", "positive"),

    # =========================================================
    # EMOJI
    # =========================================================

    ("I love this! 🔥", "positive"),
    ("Amazing! 🔥", "positive"),
    ("I hate this! 😡", "negative"),
    ("This is terrible! 😡", "negative"),
    ("I love this ❤️", "positive"),

    # =========================================================
    # CAPITALIZATION
    # =========================================================

    ("I LOVE this.", "positive"),
    ("I HATE this.", "negative"),
    ("This is AMAZING.", "positive"),
    ("This is TERRIBLE.", "negative"),
    ("This is GOOD.", "positive"),

    # =========================================================
    # MORPHOLOGY
    # =========================================================

    ("I loved this.", "positive"),
    ("I am loving this.", "positive"),
    ("She likes this.", "positive"),
    ("They liked this.", "positive"),
    ("I hated this.", "negative"),
    ("He hates this.", "negative"),
    ("I am disappointed.", "negative"),
    ("She was happier.", "positive"),
    ("This is better.", "positive"),
    ("This is worse.", "negative"),

    # =========================================================
    # ASPECT SENTIMENT
    # =========================================================

    ("The camera is amazing.", "positive"),
    ("The battery is terrible.", "negative"),
    ("The screen is excellent.", "positive"),
    ("The keyboard is bad.", "negative"),
    ("The design is beautiful.", "positive"),

    # =========================================================
    # CONTRAST
    # =========================================================

    ("I love this, but the battery is terrible.", "negative"),
    ("I hate this, but the design is amazing.", "positive"),
    ("The camera is terrible. However, the screen is amazing.", "positive"),
    ("The camera is amazing. However, the battery is terrible.", "negative"),
    ("Although the camera is bad, the screen is amazing.", "positive"),

    # =========================================================
    # MIXED
    # =========================================================

    ("I love the camera, but I hate the battery.", "mixed"),
    ("I hate the camera, but I love the battery.", "mixed"),
    ("I love the screen, but I hate the design.", "mixed"),
    ("The camera is amazing, but the battery is terrible.", "mixed"),
    ("The design is fantastic, but the keyboard is awful.", "mixed"),

    # =========================================================
    # NEUTRAL
    # =========================================================

    ("This is okay.", "neutral"),
    ("The table is made of wood.", "neutral"),
    ("The phone has a camera.", "neutral"),
    ("The battery lasts five hours.", "neutral"),
    ("The screen is six inches.", "neutral"),
]


def confidence_bucket(confidence: float) -> str:

    if confidence < 0.50:
        return "0-50%"

    if confidence < 0.60:
        return "50-60%"

    if confidence < 0.70:
        return "60-70%"

    if confidence < 0.80:
        return "70-80%"

    if confidence < 0.90:
        return "80-90%"

    if confidence < 1.00:
        return "90-100%"

    return "100%"


def main():

    print()
    print("Sentix Confidence Evaluation")
    print("=" * 90)

    correct = 0

    buckets = {
        "0-50%": {"total": 0, "correct": 0},
        "50-60%": {"total": 0, "correct": 0},
        "60-70%": {"total": 0, "correct": 0},
        "70-80%": {"total": 0, "correct": 0},
        "80-90%": {"total": 0, "correct": 0},
        "90-100%": {"total": 0, "correct": 0},
        "100%": {"total": 0, "correct": 0},
    }

    for text, expected in TEST_CASES:

        result = analyzer.predict(text)

        is_correct = (
            result.label == expected
        )

        if is_correct:
            correct += 1

        bucket = confidence_bucket(
            result.confidence
        )

        buckets[bucket]["total"] += 1

        if is_correct:
            buckets[bucket]["correct"] += 1

        status = "✓" if is_correct else "✗"

        print(
            f"{status} "
            f"{result.confidence:7.2%} "
            f"{result.label:8} "
            f"expected={expected:8} "
            f"| {text}"
        )

    total = len(TEST_CASES)

    accuracy = correct / total

    print()
    print("=" * 90)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct:  {correct}/{total}")
    print(f"Cases:    {total}")

    print()
    print("Confidence Calibration Buckets")
    print("=" * 90)

    for bucket, values in buckets.items():

        if values["total"] == 0:
            continue

        bucket_accuracy = (
            values["correct"]
            / values["total"]
        )

        print(
            f"{bucket:10} "
            f"cases={values['total']:3} "
            f"accuracy={bucket_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()