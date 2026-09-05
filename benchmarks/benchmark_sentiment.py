from collections import defaultdict
from time import perf_counter

from sentix import SentimentAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# =========================================================
# 100-CASE BALANCED BENCHMARK
# =========================================================

DATASET = [

    # -----------------------------------------------------
    # 1. Basic polarity
    # -----------------------------------------------------

    ("I love this product.", "positive", "basic"),
    ("This is amazing.", "positive", "basic"),
    ("This product is excellent.", "positive", "basic"),
    ("I really like this.", "positive", "basic"),
    ("This is fantastic.", "positive", "basic"),

    ("I hate this product.", "negative", "basic"),
    ("This is terrible.", "negative", "basic"),
    ("This product is awful.", "negative", "basic"),
    ("I dislike this.", "negative", "basic"),
    ("This is horrible.", "negative", "basic"),

    # -----------------------------------------------------
    # 2. Negation
    # -----------------------------------------------------

    ("I don't like this.", "negative", "negation"),
    ("I don't love this.", "negative", "negation"),
    ("This is not good.", "negative", "negation"),
    ("This is not bad.", "positive", "negation"),
    ("This is not terrible.", "positive", "negation"),

    ("I never liked this.", "negative", "negation"),
    ("I never hated this.", "positive", "negation"),
    ("I don't think this is good.", "negative", "negation"),
    ("I don't think this is bad.", "positive", "negation"),
    ("This isn't amazing.", "negative", "negation"),

    # -----------------------------------------------------
    # 3. Intensifiers
    # -----------------------------------------------------

    ("This is very good.", "positive", "intensity"),
    ("This is really good.", "positive", "intensity"),
    ("This is extremely good.", "positive", "intensity"),
    ("This is absolutely amazing.", "positive", "intensity"),
    ("This is incredibly fantastic.", "positive", "intensity"),

    ("This is very bad.", "negative", "intensity"),
    ("This is really terrible.", "negative", "intensity"),
    ("This is extremely awful.", "negative", "intensity"),
    ("This is absolutely horrible.", "negative", "intensity"),
    ("This is incredibly disappointing.", "negative", "intensity"),

    # -----------------------------------------------------
    # 4. Diminishers
    # -----------------------------------------------------

    ("This is slightly bad.", "negative", "diminisher"),
    ("This is somewhat bad.", "negative", "diminisher"),
    ("This is barely good.", "positive", "diminisher"),
    ("This is slightly disappointing.", "negative", "diminisher"),
    ("This is somewhat terrible.", "negative", "diminisher"),

    ("This is slightly good.", "positive", "diminisher"),
    ("This is somewhat good.", "positive", "diminisher"),
    ("I kindof like this.", "positive", "diminisher"),
    ("I kinda like this.", "positive", "diminisher"),
    ("I barely like this.", "positive", "diminisher"),

    # -----------------------------------------------------
    # 5. Contrast / clauses
    # -----------------------------------------------------

    ("I love this, but it is expensive.", "positive", "contrast"),
    ("I hate this, but the design is amazing.", "positive", "contrast"),
    ("The design is amazing, but the battery is terrible.", "mixed", "contrast"),
    ("The design is amazing, but the battery is slightly bad.", "positive", "contrast"),
    ("The design is slightly bad, but the battery is terrible.", "negative", "contrast"),

    ("I like the camera, but I hate the battery.", "mixed", "contrast"),
    ("I hate the camera, but I love the screen.", "positive", "contrast"),
    ("I dislike the design, but the screen is excellent.", "positive", "contrast"),
    ("The camera is terrible, but the battery is great.", "mixed", "contrast"),
    ("The screen is bad, but the design is fantastic.", "mixed", "contrast"),

    # -----------------------------------------------------
    # 6. Mixed sentiment
    # -----------------------------------------------------

    ("I love the camera, but I hate the battery.", "mixed", "mixed"),
    ("I hate the camera, but I love the screen.", "positive", "mixed"),
    ("The camera is terrible. The screen is amazing.", "mixed", "mixed"),
    ("I don't like the camera. I love the screen.", "mixed", "mixed"),
    ("The battery is awful. The design is beautiful.", "mixed", "mixed"),

    ("The screen is amazing. The camera is horrible.", "mixed", "mixed"),
    ("I love the design. I hate the keyboard.", "mixed", "mixed"),
    ("The keyboard is terrible. The screen is excellent.", "mixed", "mixed"),
    ("I dislike the camera. I love the battery.", "mixed", "mixed"),
    ("The design is awful. The camera is fantastic.", "mixed", "mixed"),

    # -----------------------------------------------------
    # 7. Aspect sentiment
    # -----------------------------------------------------

    ("The camera is amazing.", "positive", "aspect"),
    ("The camera is terrible.", "negative", "aspect"),
    ("The battery is excellent.", "positive", "aspect"),
    ("The battery is awful.", "negative", "aspect"),
    ("The screen is beautiful.", "positive", "aspect"),

    ("The screen is horrible.", "negative", "aspect"),
    ("The keyboard is fantastic.", "positive", "aspect"),
    ("The keyboard is disappointing.", "negative", "aspect"),
    ("The design is amazing.", "positive", "aspect"),
    ("The design is ugly.", "negative", "aspect"),

    # -----------------------------------------------------
    # 8. Morphology
    # -----------------------------------------------------

    ("I liked this product.", "positive", "morphology"),
    ("I loved this product.", "positive", "morphology"),
    ("I hated this product.", "negative", "morphology"),
    ("I am loving this product.", "positive", "morphology"),
    ("I am hating this product.", "negative", "morphology"),

    ("I was amazed by this.", "positive", "morphology"),
    ("I was disappointed by this.", "negative", "morphology"),
    ("I really liked this.", "positive", "morphology"),
    ("I really hated this.", "negative", "morphology"),
    ("I never liked this.", "negative", "morphology"),

    # -----------------------------------------------------
    # 9. Emoji / punctuation / emphasis
    # -----------------------------------------------------

    ("I love this! ❤️", "positive", "emphasis"),
    ("This is amazing! 🔥", "positive", "emphasis"),
    ("This is awesome! 😍", "positive", "emphasis"),
    ("This is terrible! 😡", "negative", "emphasis"),
    ("I hate this! 😠", "negative", "emphasis"),

    ("This is AMAZING!!!", "positive", "emphasis"),
    ("This is TERRIBLE!!!", "negative", "emphasis"),
    ("I LOVE this!!!", "positive", "emphasis"),
    ("I HATE this!!!", "negative", "emphasis"),
    ("Amazing!!! 🔥", "positive", "emphasis"),

    # -----------------------------------------------------
    # 10. Neutral / ambiguous
    # -----------------------------------------------------

    ("This is okay.", "neutral", "neutral"),
    ("This is average.", "neutral", "neutral"),
    ("The product arrived today.", "neutral", "neutral"),
    ("The camera has 12 megapixels.", "neutral", "neutral"),
    ("The battery lasts five hours.", "neutral", "neutral"),

    ("The box is on the table.", "neutral", "neutral"),
    ("I bought this yesterday.", "neutral", "neutral"),
    ("The screen is 6 inches.", "neutral", "neutral"),
    ("The product comes with a charger.", "neutral", "neutral"),
    ("This device weighs 500 grams.", "neutral", "neutral"),
]


# =========================================================
# ANALYZERS
# =========================================================

sentix = SentimentAnalyzer()
vader = SentimentIntensityAnalyzer()


# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def predict_sentix(text: str) -> str:
    return sentix.predict(text)["label"]


def predict_vader(text: str) -> str:
    score = vader.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"

    if score <= -0.05:
        return "negative"

    return "neutral"


def predict_textblob(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.05:
        return "positive"

    if polarity < -0.05:
        return "negative"

    return "neutral"


# =========================================================
# METRICS
# =========================================================

LABELS = ["positive", "negative", "neutral", "mixed"]


def calculate_metrics(results, system_index):
    """
    Calculate macro precision, recall and F1.
    """

    precisions = []
    recalls = []
    f1_scores = []

    for label in LABELS:

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for result in results:

            expected = result[1]
            predicted = result[system_index]

            if predicted == label and expected == label:
                true_positive += 1

            elif predicted == label and expected != label:
                false_positive += 1

            elif predicted != label and expected == label:
                false_negative += 1

        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = (
                true_positive /
                (true_positive + false_positive)
            )

        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = (
                true_positive /
                (true_positive + false_negative)
            )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall /
                (precision + recall)
            )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1_scores) / len(f1_scores),
    )


# =========================================================
# RUN BENCHMARK
# =========================================================

results = []

for text, expected, category in DATASET:

    sentix_result = predict_sentix(text)
    vader_result = predict_vader(text)
    textblob_result = predict_textblob(text)

    results.append(
        (
            text,
            expected,
            category,
            sentix_result,
            vader_result,
            textblob_result,
        )
    )


# =========================================================
# ACCURACY
# =========================================================

total = len(results)

systems = {
    "Sentix": 3,
    "VADER": 4,
    "TextBlob": 5,
}

accuracies = {}

for name, index in systems.items():

    correct = sum(
        result[1] == result[index]
        for result in results
    )

    accuracies[name] = (
        correct,
        correct / total
    )


# =========================================================
# PRINT OVERALL RESULTS
# =========================================================

print()
print("=" * 80)
print("SENTIX vs VADER vs TEXTBLOB")
print("=" * 80)

print(f"Dataset size: {total}")

print()

for name, (correct, accuracy) in accuracies.items():

    print(
        f"{name:<10}: "
        f"{correct}/{total} "
        f"({accuracy:.2%})"
    )


# =========================================================
# PRECISION / RECALL / F1
# =========================================================

print()
print("=" * 80)
print("MACRO METRICS")
print("=" * 80)

for name, index in systems.items():

    precision, recall, f1 = calculate_metrics(
        results,
        index
    )

    print()
    print(name)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")


# =========================================================
# CATEGORY ACCURACY
# =========================================================

print()
print("=" * 80)
print("CATEGORY ACCURACY")
print("=" * 80)

categories = defaultdict(list)

for result in results:
    categories[result[2]].append(result)

for category, category_results in categories.items():

    print()
    print(category.upper())

    for name, index in systems.items():

        correct = sum(
            result[1] == result[index]
            for result in category_results
        )

        accuracy = correct / len(category_results)

        print(
            f"  {name:<10}: "
            f"{correct}/{len(category_results)} "
            f"({accuracy:.2%})"
        )


# =========================================================
# CONFUSION MATRICES
# =========================================================

for name, index in systems.items():

    print()
    print("=" * 80)
    print(f"{name.upper()} CONFUSION MATRIX")
    print("=" * 80)

    matrix = {
        actual: {
            predicted: 0
            for predicted in LABELS
        }
        for actual in LABELS
    }

    for result in results:

        actual = result[1]
        predicted = result[index]

        matrix[actual][predicted] += 1

    print()
    print(
        f"{'Actual':<12}"
        + "".join(
            f"{label:<12}"
            for label in LABELS
        )
    )

    for actual in LABELS:

        print(
            f"{actual:<12}"
            + "".join(
                f"{matrix[actual][predicted]:<12}"
                for predicted in LABELS
            )
        )


# =========================================================
# WRONG PREDICTIONS
# =========================================================

for name, index in systems.items():

    print()
    print("=" * 80)
    print(f"{name.upper()} WRONG PREDICTIONS")
    print("=" * 80)

    wrong_count = 0

    for result in results:

        text = result[0]
        expected = result[1]
        predicted = result[index]

        if expected != predicted:

            wrong_count += 1

            print()
            print(f"Text:     {text}")
            print(f"Expected: {expected}")
            print(f"Got:      {predicted}")

    if wrong_count == 0:
        print()
        print("No incorrect predictions.")


# =========================================================
# RUNTIME
# =========================================================

print()
print("=" * 80)
print("RUNTIME")
print("=" * 80)

texts = [item[0] for item in DATASET]


start = perf_counter()

for text in texts:
    predict_sentix(text)

sentix_time = perf_counter() - start


start = perf_counter()

for text in texts:
    predict_vader(text)

vader_time = perf_counter() - start


start = perf_counter()

for text in texts:
    predict_textblob(text)

textblob_time = perf_counter() - start


print(f"Sentix   : {sentix_time:.6f} seconds")
print(f"VADER    : {vader_time:.6f} seconds")
print(f"TextBlob : {textblob_time:.6f} seconds")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)