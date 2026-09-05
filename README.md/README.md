# Sentix

A lightweight and explainable rule-based sentiment analysis library for Python.

Sentix provides sentiment classification with support for:

- Positive sentiment
- Negative sentiment
- Neutral sentiment
- Mixed sentiment
- Negation handling
- Intensity modifiers
- Diminishers
- Contrast detection
- Aspect-based sentiment
- Emotion detection
- Confidence scoring
- Evidence-based explanations

## Installation

```bash
pip install sentix
```

## Quick Start

```python
from sentix import SentimentAnalyzer

analyzer = SentimentAnalyzer()

result = analyzer.predict(
    "I love the camera, but I hate the battery."
)

print(result.label)
print(result.score)
print(result.confidence)
```

Example output:

```text
mixed
0.0
0.49
```

## Full Result

```python
from sentix import SentimentAnalyzer

analyzer = SentimentAnalyzer()

result = analyzer.predict(
    "The camera is amazing, but the battery is terrible."
)

print(result.to_dict())
```

Example structure:

```python
{
    "label": "mixed",
    "score": 0.0,
    "positive": 0.5,
    "negative": 0.5,
    "neutral": 0.0,
    "confidence": 0.49,
    "emotions": {},
    "aspects": {},
    "evidence": []
}
```

## Supported Sentiment Labels

Sentix can return four sentiment labels:

| Label | Meaning |
|---|---|
| `positive` | Overall positive sentiment |
| `negative` | Overall negative sentiment |
| `neutral` | No significant sentiment detected |
| `mixed` | Both positive and negative sentiment detected |

## Negation

Sentix handles common negation patterns.

```python
analyzer.predict("I don't like this.")
```

Expected sentiment:

```text
negative
```

Example:

```python
analyzer.predict("I don't think this is good.")
```

Expected sentiment:

```text
negative
```

## Intensity

Sentix supports intensity modifiers.

```python
analyzer.predict(
    "This is extremely amazing."
)
```

The word `extremely` increases the strength of the sentiment.

## Diminishers

Sentix also supports diminishing modifiers.

```python
analyzer.predict(
    "The battery is slightly bad."
)
```

The negative sentiment is weaker than:

```python
"The battery is terrible."
```

## Contrast Handling

Sentix analyzes contrast expressions such as `but`.

```python
analyzer.predict(
    "I love this, but the battery is terrible."
)
```

Contrast analysis considers sentiment before and after the contrast marker.

## Mixed Sentiment

Sentix can detect conflicting sentiment.

```python
analyzer.predict(
    "I love the screen, but I hate the battery."
)
```

Possible output:

```text
mixed
```

## Aspect-Based Sentiment

Sentix can identify sentiment associated with specific aspects.

```python
result = analyzer.predict(
    "The camera is amazing but the battery is terrible."
)

print(result.aspects)
```

Example:

```python
{
    "camera": {
        "sentiment": "positive",
        "score": 2.0
    },
    "battery": {
        "sentiment": "negative",
        "score": -2.0
    }
}
```

## Emotion Detection

Sentix can detect emotion-related signals.

```python
result = analyzer.predict(
    "I am extremely happy and excited."
)

print(result.emotions)
```

## Evidence

Sentix provides explainable sentiment evidence.

```python
result = analyzer.predict(
    "This product is extremely amazing!"
)

for item in result.evidence:
    print(item)
```

Evidence may include:

- sentiment word
- base score
- intensity modifier
- negation modifier
- capitalization modifier
- final score

## Benchmark

Current benchmark results on the Sentix benchmark dataset:

| Library | Accuracy | Macro F1 |
|---|---:|---:|
| **Sentix** | **97%** | **0.9632** |
| VADER | 83% | 0.6793 |
| TextBlob | 69% | 0.5264 |

### Category Performance

| Category | Sentix |
|---|---:|
| Basic | 100% |
| Negation | 100% |
| Intensity | 100% |
| Diminisher | 100% |
| Contrast | 80% |
| Mixed | 90% |
| Aspect | 100% |
| Morphology | 100% |
| Emphasis | 100% |
| Neutral | 100% |

## Testing

Run the complete test suite:

```bash
python -m pytest
```

Current status:

```text
173 passed
```

## Benchmark

Run:

```bash
python benchmarks/benchmark_sentiment.py
```

## Performance

On the current benchmark dataset of 100 samples:

```text
Sentix: approximately 0.036 seconds
```

This is approximately:

```text
0.36 milliseconds per sample
```

Performance will vary depending on hardware and Python version.

## Design Philosophy

Sentix focuses on:

- Lightweight implementation
- Explainability
- Rule-based reasoning
- No external machine learning model requirement
- Fast inference
- Transparent sentiment evidence

## License

MIT License

## Author

Sentix Contributors