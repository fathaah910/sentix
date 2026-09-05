from dataclasses import dataclass
from typing import Any

from .explanation import (
    SentimentEvidence,
    explain_result,
)


@dataclass
class SentimentResult:
    """
    Structured result returned by Sentix.
    """

    label: str
    score: float
    positive: float
    negative: float
    neutral: float
    confidence: float
    emotions: dict[str, float]
    aspects: dict[str, dict[str, Any]]
    evidence: list[SentimentEvidence]

    # Explanation metadata
    raw_score: float | None = None
    punctuation_modifier: float = 1.0

    def __getitem__(self, key: str) -> Any:

        if not isinstance(key, str):
            raise KeyError(key)

        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: str) -> bool:

        return key in {
            "label",
            "score",
            "positive",
            "negative",
            "neutral",
            "confidence",
            "emotions",
            "aspects",
            "evidence",
            "raw_score",
            "punctuation_modifier",
        }

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:

        if key in self:
            return self[key]

        return default

    def to_dict(self) -> dict[str, Any]:

        return {
            "label": self.label,
            "score": self.score,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "confidence": self.confidence,
            "emotions": self.emotions,
            "aspects": self.aspects,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "raw_score": self.raw_score,
            "punctuation_modifier": (
                self.punctuation_modifier
            ),
        }

    def explain(
        self,
        output_format: str = "text",
    ) -> str | dict[str, Any]:

        return explain_result(
            label=self.label,
            score=self.score,
            confidence=self.confidence,
            evidence=self.evidence,
            output_format=output_format,
            raw_score=self.raw_score,
            punctuation_modifier=(
                self.punctuation_modifier
            ),
        )