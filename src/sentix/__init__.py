"""
Sentix
======

A lightweight rule-based sentiment analysis library for Python.
"""

from .analyzer import SentimentAnalyzer
from .result import SentimentResult


__version__ = "1.0.0"


__all__ = [
    "SentimentAnalyzer",
    "SentimentResult",
]