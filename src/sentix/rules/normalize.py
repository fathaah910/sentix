import math


def normalize_score(score: float) -> float:
    return score / (abs(score) + 1.0)