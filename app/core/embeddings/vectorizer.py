"""Pure-Python TF-IDF vectorisation and cosine similarity.

Deliberately dependency-free. A resume/JD pair is two short documents, so the
cost of a heavyweight embedding model (download size, cold start, GPU, an
outbound network call per request) buys very little over well-tuned lexical
similarity, and lexical similarity is *explainable* - we can show the exact
terms that drove the score, which a dense vector cannot.

IDF is estimated from the bullets and sentences of the documents under
comparison, so common resume boilerplate is down-weighted automatically.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from app.core.parsing.normalize import content_tokens, is_bullet, strip_bullet

#: Bigrams let "machine learning" outrank the sum of "machine" and "learning".
BIGRAM_WEIGHT = 0.6

Vector = dict[str, float]


def _terms(text: str) -> list[str]:
    unigrams = content_tokens(text)
    bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:], strict=False)]
    return unigrams + bigrams


def _weight(term: str) -> float:
    return BIGRAM_WEIGHT if "_" in term else 1.0


def split_units(text: str) -> list[str]:
    """Break a document into the pseudo-documents used for IDF estimation."""
    units: list[str] = []
    for line in text.split("\n"):
        stripped = strip_bullet(line) if is_bullet(line) else line.strip()
        if len(stripped) >= 15:
            units.append(stripped)
    return units or ([text.strip()] if text.strip() else [])


def build_idf(documents: Sequence[str], *, smooth: float = 1.0) -> dict[str, float]:
    """Smoothed inverse document frequency over a corpus of pseudo-documents."""
    total = len(documents)
    if total == 0:
        return {}
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(_terms(document)))
    return {
        term: math.log((total + smooth) / (freq + smooth)) + 1.0
        for term, freq in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> Vector:
    """L2-normalised sublinear TF-IDF vector.

    Sublinear TF (``1 + log(tf)``) stops a resume that repeats "Python" nine
    times from scoring as nine times more relevant than one that says it twice.
    """
    counts = Counter(_terms(text))
    if not counts:
        return {}
    vector: Vector = {}
    for term, count in counts.items():
        weight = (1.0 + math.log(count)) * idf.get(term, 1.0) * _weight(term)
        if weight > 0:
            vector[term] = weight
    norm = math.sqrt(sum(v * v for v in vector.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two L2-normalised vectors, clamped to [0, 1]."""
    if not a or not b:
        return 0.0
    # Iterate the smaller vector: resume/JD vocabularies differ a lot in size.
    if len(a) > len(b):
        a, b = b, a
    total = sum(value * b.get(term, 0.0) for term, value in a.items())
    return max(0.0, min(1.0, total))


def similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two documents, with IDF derived from both."""
    corpus = split_units(text_a) + split_units(text_b)
    if not corpus:
        return 0.0
    idf = build_idf(corpus)
    return cosine(vectorize(text_a, idf), vectorize(text_b, idf))


def top_overlapping_terms(
    text_a: str, text_b: str, limit: int = 12
) -> list[tuple[str, float]]:
    """The shared terms contributing most to the similarity score.

    This is what makes the semantic score explainable in the UI.
    """
    corpus = split_units(text_a) + split_units(text_b)
    if not corpus:
        return []
    idf = build_idf(corpus)
    vector_a = vectorize(text_a, idf)
    vector_b = vectorize(text_b, idf)
    contributions = [
        (term.replace("_", " "), round(value * vector_b[term], 5))
        for term, value in vector_a.items()
        if term in vector_b
    ]
    contributions.sort(key=lambda pair: -pair[1])
    return contributions[:limit]


def missing_terms(source: str, target: str, limit: int = 15) -> list[str]:
    """High-IDF terms present in ``target`` (the JD) but absent from ``source``."""
    corpus = split_units(source) + split_units(target)
    if not corpus:
        return []
    idf = build_idf(corpus)
    source_terms = set(_terms(source))
    target_vector = vectorize(target, idf)
    absent = [
        (term, weight)
        for term, weight in target_vector.items()
        if term not in source_terms and "_" not in term and len(term) > 2
    ]
    absent.sort(key=lambda pair: -pair[1])
    return [term for term, _ in absent[:limit]]


def centroid(vectors: Iterable[Vector]) -> Vector:
    """Mean of several vectors, re-normalised. Used for multi-JD matching."""
    total: Vector = {}
    count = 0
    for vector in vectors:
        count += 1
        for term, value in vector.items():
            total[term] = total.get(term, 0.0) + value
    if not count or not total:
        return {}
    norm = math.sqrt(sum(v * v for v in total.values()))
    return {term: value / norm for term, value in total.items()} if norm else {}
