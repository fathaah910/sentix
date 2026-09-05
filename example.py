from sentix import SentimentAnalyzer


analyzer = SentimentAnalyzer()

texts = [
    "I love this product!",
    "This is terrible.",
    "The product is okay.",
    "I don't like this.",
    "This is absolutely amazing! 🔥",
]

for text in texts:
    result = analyzer.predict(text)

    print(text)
    print(result)
    print()