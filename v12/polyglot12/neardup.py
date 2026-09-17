"""Near-duplicate filter: character-3-gram Jaccard similarity on normalized text.

Candidate pairs come from an inverted index over rare 3-grams, then exact Jaccard is computed, so
50k x 1k comparisons stay fast. A new text is dropped when Jaccard > threshold (0.6) with ANY
protected (validation/test/audit/contrast) message.
"""
from collections import defaultdict

from .hashing import normalize


def grams(text):
    try:
        s = normalize(text)
    except ValueError:
        s = ' '.join(text.lower().split())
    s = f' {s} '
    return frozenset(s[i:i + 3] for i in range(len(s) - 2))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


class NearDupIndex:
    def __init__(self, protected_texts):
        self.items = [(t, grams(t)) for t in protected_texts]
        self.index = defaultdict(list)
        for i, (_, g) in enumerate(self.items):
            for x in g:
                self.index[x].append(i)

    def best_match(self, text):
        g = grams(text)
        if not g:
            return 0.0, None
        # Exact |A∩B| per candidate from the inverted index, then Jaccard = |A∩B| / (|A|+|B|-|A∩B|).
        counts = defaultdict(int)
        for x in g:
            for i in self.index.get(x, ()):
                counts[i] += 1
        best, best_text = 0.0, None
        for i, shared in counts.items():
            other = self.items[i][1]
            jac = shared / max(1, len(g) + len(other) - shared)
            if jac > best:
                best, best_text = jac, self.items[i][0]
        return best, best_text

    def is_near_dup(self, text, threshold=0.6):
        return self.best_match(text)[0] > threshold
