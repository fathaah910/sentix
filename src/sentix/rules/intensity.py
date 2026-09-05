INTENSIFIERS = {
    "very": 1.3,
    "really": 1.2,
    "extremely": 1.6,
    "absolutely": 1.7,
    "incredibly": 1.6,
    "insanely": 1.8,
}

DIMINISHERS = {
    "slightly": 0.5,
    "somewhat": 0.7,
    "barely": 0.4,
    "kindof": 0.7,
    "kinda": 0.7,
}


def get_intensity_modifier(tokens: list[str], index: int) -> float:
    if index == 0:
        return 1.0

    previous = tokens[index - 1].lower()

    if previous in INTENSIFIERS:
        return INTENSIFIERS[previous]

    if previous in DIMINISHERS:
        return DIMINISHERS[previous]

    return 1.0