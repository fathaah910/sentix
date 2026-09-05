"""
Explainability utilities for Sentix.
"""

from dataclasses import dataclass
from typing import Any
import json


@dataclass
class SentimentEvidence:
    """
    Represents one piece of sentiment evidence.
    """

    text: str
    score: float
    source: str

    # General modifier
    modifier: float = 1.0

    # Original sentiment information
    base_score: float | None = None

    # Rule modifiers
    intensity_modifier: float = 1.0
    capitalization_modifier: float = 1.0
    negation_modifier: float = 1.0

    # Negation information
    negated: bool = False
    negation_count: int = 0

    # Conditional information
    conditional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert evidence into a dictionary.
        """

        return {
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "modifier": self.modifier,
            "base_score": self.base_score,
            "intensity_modifier": (
                self.intensity_modifier
            ),
            "capitalization_modifier": (
                self.capitalization_modifier
            ),
            "negation_modifier": (
                self.negation_modifier
            ),
            "negated": self.negated,
            "negation_count": (
                self.negation_count
            ),
            "conditional": self.conditional,
        }


def format_evidence(
    evidence: list[SentimentEvidence],
) -> list[str]:
    """
    Convert evidence objects into human-readable lines.
    """

    lines: list[str] = []

    for item in evidence:

        # =====================================================
        # Lexicon
        # =====================================================

        if item.source == "lexicon":

            lines.append(
                f'  • "{item.text}"'
            )

            if item.base_score is not None:

                lines.append(
                    f"      Base score          : "
                    f"{item.base_score:+.2f}"
                )

            if item.intensity_modifier != 1.0:

                lines.append(
                    f"      Intensity modifier   : "
                    f"×{item.intensity_modifier:.2f}"
                )

            if item.capitalization_modifier != 1.0:

                lines.append(
                    f"      Capitalization      : "
                    f"×{item.capitalization_modifier:.2f}"
                )

            if item.negated:

                if item.negation_count > 1:

                    lines.append(
                        f"      Negation            : "
                        f"{item.negation_count} "
                        f"negations "
                        f"→ ×{item.negation_modifier:.2f}"
                    )

                else:

                    lines.append(
                        f"      Negation            : "
                        f"×{item.negation_modifier:.2f}"
                    )

            else:

                lines.append(
                    "      Negation            : No"
                )

            if item.conditional:

                lines.append(
                    "      Conditional         : Yes"
                )

            else:

                lines.append(
                    "      Conditional         : No"
                )

            lines.append(
                f"      Final contribution  : "
                f"{item.score:+.2f}"
            )

        # =====================================================
        # Emoji
        # =====================================================

        elif item.source == "emoji":

            lines.append(
                f"  • {item.text}"
            )

            lines.append(
                "      Source              : emoji"
            )

            if item.base_score is not None:

                lines.append(
                    f"      Base score          : "
                    f"{item.base_score:+.2f}"
                )

            lines.append(
                f"      Contribution        : "
                f"{item.score:+.2f}"
            )

        # =====================================================
        # Phrase
        # =====================================================

        elif item.source == "phrase":

            lines.append(
                f'  • "{item.text}"'
            )

            lines.append(
                "      Source              : phrase"
            )

            if item.base_score is not None:

                lines.append(
                    f"      Base score          : "
                    f"{item.base_score:+.2f}"
                )

            if item.conditional:

                lines.append(
                    "      Conditional         : Yes"
                )

            lines.append(
                f"      Contribution        : "
                f"{item.score:+.2f}"
            )

        # =====================================================
        # Punctuation
        # =====================================================

        elif item.source == "punctuation":

            lines.append(
                f"  • {item.text}"
            )

            lines.append(
                "      Source              : punctuation"
            )

            lines.append(
                f"      Modifier             : "
                f"×{item.modifier:.2f}"
            )

        # =====================================================
        # Context
        # =====================================================

        elif item.source == "context":

            lines.append(
                f'  • "{item.text}"'
            )

            lines.append(
                "      Source              : context"
            )

            if item.base_score is not None:

                lines.append(
                    f"      Context score       : "
                    f"{item.base_score:+.2f}"
                )

            lines.append(
                f"      Contribution        : "
                f"{item.score:+.2f}"
            )

        # =====================================================
        # Unknown/custom evidence
        # =====================================================

        else:

            lines.append(
                f'  • "{item.text}"'
            )

            lines.append(
                f"      Source              : "
                f"{item.source}"
            )

            if item.base_score is not None:

                lines.append(
                    f"      Base score          : "
                    f"{item.base_score:+.2f}"
                )

            lines.append(
                f"      Contribution        : "
                f"{item.score:+.2f}"
            )

    return lines


def explain_result(
    label: str,
    score: float,
    confidence: float,
    evidence: list[SentimentEvidence],
    output_format: str = "text",
    raw_score: float | None = None,
    punctuation_modifier: float = 1.0,
) -> str | dict[str, Any]:
    """
    Generate a human-readable or structured explanation.

    Supported output formats:

        text
        dict
        json
    """

    if output_format not in {
        "text",
        "dict",
        "json",
    }:

        raise ValueError(
            "output_format must be "
            "'text', 'dict', or 'json'"
        )

    if raw_score is None:
        raw_score = score

    evidence_data = [
        item.to_dict()
        for item in evidence
    ]

    data: dict[str, Any] = {
        "sentiment": label,
        "raw_score": raw_score,
        "punctuation_modifier": (
            punctuation_modifier
        ),
        "score": score,
        "confidence": confidence,
        "evidence": evidence_data,
    }

    # =========================================================
    # Dictionary
    # =========================================================

    if output_format == "dict":
        return data

    # =========================================================
    # JSON
    # =========================================================

    if output_format == "json":

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================
    # Human-readable text
    # =========================================================

    lines = [
        "Sentix Analysis",
        "────────────────────────────",
        "",
        f"Sentiment : {label.upper()}",
        f"Raw score : {raw_score:.2f}",
    ]

    if punctuation_modifier != 1.0:

        lines.append(
            f"Punctuation: "
            f"×{punctuation_modifier:.2f}"
        )

        lines.append(
            f"Final score: "
            f"{raw_score:.2f} × "
            f"{punctuation_modifier:.2f} "
            f"= {score:.2f}"
        )

    else:

        lines.append(
            f"Final score: {score:.2f}"
        )

    lines.extend([
        f"Confidence: {confidence:.2%}",
        "",
        "Evidence:",
    ])

    evidence_lines = format_evidence(
        evidence
    )

    if evidence_lines:

        lines.extend(
            evidence_lines
        )

    else:

        lines.append(
            "  • No sentiment evidence found"
        )

    return "\n".join(lines)