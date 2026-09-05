from sentix import SentimentAnalyzer


analyzer = SentimentAnalyzer()


TEST_CASES = [

    # =========================================================
    # UNCERTAIN POSITIVE
    # =========================================================

    ("Maybe this is good.", "positive"),
    ("Perhaps this is amazing.", "positive"),
    ("This might be great.", "positive"),
    ("This could be excellent.", "positive"),
    ("I guess I like this.", "positive"),
    ("I think I love this.", "positive"),
    ("This seems nice.", "positive"),

    # =========================================================
    # UNCERTAIN NEGATIVE
    # =========================================================

    ("Maybe this is bad.", "negative"),
    ("Perhaps this is terrible.", "negative"),
    ("This might be awful.", "negative"),
    ("This could be horrible.", "negative"),
    ("I guess I hate this.", "negative"),
    ("I think I dislike this.", "negative"),

    # =========================================================
    # HEDGING + CONTRADICTION
    # =========================================================

    ("Maybe this is amazing, but I'm not sure.", "positive"),
    ("I think this is good, but maybe not.", "positive"),
    ("It seems great, although I'm uncertain.", "positive"),
    ("I might love this, but I'm unsure.", "positive"),

    # =========================================================
    # MIXED / CONFLICTING
    # =========================================================

    ("I love the camera, but I hate the battery.", "mixed"),
    ("The screen is amazing, but the battery is terrible.", "mixed"),
    ("I hate the design, but I love the camera.", "mixed"),
    ("The camera is terrible, but the screen is fantastic.", "mixed"),

    # =========================================================
    # CONTRAST WITH DIFFERENT STRENGTHS
    # =========================================================

    ("I love this, but the battery is slightly bad.", "positive"),
    ("I hate this, but the design is slightly good.", "negative"),
    ("The camera is terrible, but the screen is extremely amazing.", "positive"),
    ("The camera is extremely terrible, but the screen is good.", "negative"),

    # =========================================================
    # NEGATION + UNCERTAINTY
    # =========================================================

    ("Maybe this is not good.", "negative"),
    ("Perhaps this is not bad.", "positive"),
    ("I don't think this is amazing.", "negative"),
    ("I don't think this is bad.", "positive"),
    ("I might not like this.", "negative"),

    # =========================================================
    # DOUBLE NEGATION / COMPLEX LANGUAGE
    # =========================================================

    ("I don't think this is not good.", "positive"),
    ("I can't say that I don't like this.", "positive"),
    ("It's not that I hate this.", "positive"),

    # =========================================================
    # SARCASM-LIKE LANGUAGE
    #
    # These are intentionally difficult for a rule-based
    # sentiment system.
    # =========================================================

    ("Yeah, great job breaking everything.", "negative"),
    ("Fantastic, another error.", "negative"),
    ("Wonderful, the app crashed again.", "negative"),
    ("Amazing, it stopped working.", "negative"),
    ("Perfect, another problem.", "negative"),

    # =========================================================
    # POSITIVE WORD + NEGATIVE CONTEXT
    # =========================================================

    ("I love how badly this works.", "negative"),
    ("Amazing how terrible this product is.", "negative"),
    ("Great, now everything is broken.", "negative"),
    ("I like the fact that it keeps crashing.", "negative"),

    # =========================================================
    # NEGATIVE WORD + POSITIVE CONTEXT
    # =========================================================

    ("The terrible wait was completely worth it.", "positive"),
    ("The bad experience ended beautifully.", "positive"),
    ("It was awful at first, but now it's amazing.", "positive"),

    # =========================================================
    # TEMPORAL CHANGE
    # =========================================================

    ("At first I loved it, but now I hate it.", "negative"),
    ("At first I hated it, but now I love it.", "positive"),
    ("I loved it yesterday, but today I hate it.", "negative"),
    ("I hated it yesterday, but today I love it.", "positive"),

    # =========================================================
    # CONDITIONAL SENTIMENT
    # =========================================================

    ("If the battery were better, I would love this.", "negative"),
    ("If the camera were amazing, I would like this.", "negative"),
    ("I would love this if the screen were better.", "negative"),
    ("If this were cheaper, it would be great.", "negative"),

    # =========================================================
    # ASPECT CONFLICT
    # =========================================================

    ("The camera is amazing, but the battery is terrible.", "mixed"),
    ("The screen is beautiful, but the design is ugly.", "mixed"),
    ("The keyboard is excellent, but the screen is awful.", "mixed"),
    ("The design is fantastic, but the camera is horrible.", "mixed"),

    # =========================================================
    # WEAK SENTIMENT
    # =========================================================

    ("I slightly like this.", "positive"),
    ("I somewhat like this.", "positive"),
    ("I barely like this.", "positive"),
    ("I slightly dislike this.", "negative"),
    ("I somewhat dislike this.", "negative"),

    # =========================================================
    # NEUTRAL / AMBIGUOUS
    # =========================================================

    ("I received the product yesterday.", "neutral"),
    ("The product arrived today.", "neutral"),
    ("The phone has a camera.", "neutral"),
    ("The battery lasts five hours.", "neutral"),
    ("The screen is six inches.", "neutral"),
]


def confidence_level(confidence: float) -> str:

    if confidence < 0.40:
        return "very-low"

    if confidence < 0.60:
        return "low"

    if confidence < 0.80:
        return "moderate"

    if confidence < 0.95:
        return "high"

    return "very-high"


def main():

    print()
    print("Sentix Adversarial Confidence Evaluation")
    print("=" * 100)

    correct = 0

    results = []

    for text, expected in TEST_CASES:

        result = analyzer.predict(text)

        is_correct = (
            result.label == expected
        )

        if is_correct:
            correct += 1

        results.append(
            {
                "text": text,
                "expected": expected,
                "predicted": result.label,
                "confidence": result.confidence,
                "correct": is_correct,
            }
        )

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
    print("=" * 100)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct:  {correct}/{total}")
    print(f"Cases:    {total}")

    # =========================================================
    # Incorrect predictions
    # =========================================================

    incorrect = [
        item
        for item in results
        if not item["correct"]
    ]

    print()
    print("Incorrect Predictions")
    print("=" * 100)

    if not incorrect:
        print("None")

    else:
        for item in incorrect:

            print(
                f"{item['confidence']:7.2%} "
                f"predicted={item['predicted']:8} "
                f"expected={item['expected']:8} "
                f"| {item['text']}"
            )

    # =========================================================
    # High-confidence mistakes
    # =========================================================

    high_confidence_errors = [
        item
        for item in incorrect
        if item["confidence"] >= 0.80
    ]

    print()
    print("High-Confidence Errors")
    print("=" * 100)

    if not high_confidence_errors:
        print("None")

    else:
        for item in high_confidence_errors:

            print(
                f"{item['confidence']:7.2%} "
                f"predicted={item['predicted']:8} "
                f"expected={item['expected']:8} "
                f"| {item['text']}"
            )

    # =========================================================
    # Confidence summary
    # =========================================================

    print()
    print("Confidence Summary")
    print("=" * 100)

    levels = {
        "very-low": [],
        "low": [],
        "moderate": [],
        "high": [],
        "very-high": [],
    }

    for item in results:

        level = confidence_level(
            item["confidence"]
        )

        levels[level].append(item)

    for level, items in levels.items():

        if not items:
            continue

        level_correct = sum(
            item["correct"]
            for item in items
        )

        level_accuracy = (
            level_correct / len(items)
        )

        average_confidence = (
            sum(
                item["confidence"]
                for item in items
            )
            / len(items)
        )

        print(
            f"{level:10} "
            f"cases={len(items):3} "
            f"avg_confidence="
            f"{average_confidence:.2%} "
            f"accuracy="
            f"{level_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()