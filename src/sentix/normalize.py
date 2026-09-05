"""
Lexical normalization for Sentix.

This module converts common grammatical and morphological
forms into a sentiment-lexicon-compatible form.

The normalizer is intentionally lightweight and dependency-free.
It uses explicit irregular mappings plus conservative suffix rules.
"""

import re


# ============================================================
# IRREGULAR / EXPLICIT FORMS
# ============================================================

IRREGULAR_FORMS = {

    # --------------------------------------------------------
    # LOVE
    # --------------------------------------------------------

    "loved": "love",
    "loving": "love",
    "loves": "love",

    # --------------------------------------------------------
    # LIKE
    # --------------------------------------------------------

    "liked": "like",
    "liking": "like",
    "likes": "like",

    # --------------------------------------------------------
    # HATE
    # --------------------------------------------------------

    "hated": "hate",
    "hating": "hate",
    "hates": "hate",

    # --------------------------------------------------------
    # DISLIKE
    # --------------------------------------------------------

    "disliked": "dislike",
    "disliking": "dislike",
    "dislikes": "dislike",

    # --------------------------------------------------------
    # AMAZE
    # --------------------------------------------------------

    "amazed": "amazing",
    "amaze": "amazing",
    "amazes": "amazing",
    "amazing": "amazing",

    # --------------------------------------------------------
    # DISAPPOINT
    # --------------------------------------------------------

    "disappointed": "disappointing",
    "disappoint": "disappointing",
    "disappoints": "disappointing",
    "disappointing": "disappointing",

    # --------------------------------------------------------
    # COMMON SENTIMENT FORMS
    # --------------------------------------------------------

    "happier": "happy",
    "happiest": "happy",

    "sadder": "sad",
    "saddest": "sad",

    "uglier": "ugly",
    "ugliest": "ugly",

    "better": "good",
    "best": "best",

    "worse": "bad",
    "worst": "worst",
}


# ============================================================
# CONSERVATIVE SUFFIX RULES
# ============================================================

def _remove_ing(word: str) -> str:
    """
    Convert a simple -ing form into a base form.

    Examples:
        loving   -> lov
        liking   -> lik
        hating   -> hat

    This function is only a fallback.
    Explicit sentiment forms are handled above.
    """

    if not word.endswith("ing"):
        return word

    if len(word) <= 5:
        return word

    return word[:-3]


def _remove_ed(word: str) -> str:
    """
    Convert a simple -ed form into a base form.

    This is a conservative fallback and does not replace
    the explicit irregular mappings.
    """

    if not word.endswith("ed"):
        return word

    if len(word) <= 4:
        return word

    return word[:-2]


def _remove_plural_s(word: str) -> str:
    """
    Convert a simple plural/present-tense form into a base form.
    """

    if not word.endswith("s"):
        return word

    if len(word) <= 3:
        return word

    return word[:-1]


# ============================================================
# PUBLIC NORMALIZER
# ============================================================

def normalize_word(word: str) -> str:
    """
    Normalize a word into a sentiment-lexicon-compatible form.

    Examples:
        loved      -> love
        loves      -> love
        disliked   -> dislike
        disliking  -> dislike
        hated      -> hate
        likes      -> like

    Parameters
    ----------
    word:
        Input token.

    Returns
    -------
    str
        Normalized lowercase word.
    """

    normalized = word.lower().strip()

    if not normalized:
        return normalized

    # Remove accidental surrounding punctuation.
    normalized = re.sub(
        r"^[^\w]+|[^\w]+$",
        "",
        normalized,
    )

    if not normalized:
        return normalized

    # --------------------------------------------------------
    # Explicit mappings have priority.
    # --------------------------------------------------------

    if normalized in IRREGULAR_FORMS:
        return IRREGULAR_FORMS[normalized]

    # --------------------------------------------------------
    # Conservative fallback rules.
    #
    # These are intentionally applied only when an explicit
    # mapping does not already exist.
    # --------------------------------------------------------

    if normalized.endswith("ies") and len(normalized) > 4:
        candidate = normalized[:-3] + "y"

        if candidate in IRREGULAR_FORMS:
            return IRREGULAR_FORMS[candidate]

    if normalized.endswith("ing"):
        candidate = _remove_ing(normalized)

        if candidate in IRREGULAR_FORMS:
            return IRREGULAR_FORMS[candidate]

        return candidate

    if normalized.endswith("ed"):
        candidate = _remove_ed(normalized)

        if candidate in IRREGULAR_FORMS:
            return IRREGULAR_FORMS[candidate]

        return candidate

    if normalized.endswith("s"):
        candidate = _remove_plural_s(normalized)

        if candidate in IRREGULAR_FORMS:
            return IRREGULAR_FORMS[candidate]

        return candidate

    return normalized