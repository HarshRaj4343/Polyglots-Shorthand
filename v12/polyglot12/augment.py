"""Meaning-preserving character noise for consistency training and KD augmentation.

Rules (checked by tests):
- tokens are whitespace-split; only purely alphabetic ASCII tokens of length >= 3 are edited;
- negators / negation-scope words and every non-alphabetic token (emoji, punctuation, numbers)
  are never changed, and tokens are never inserted, deleted or reordered;
- an edit is reverted if it produces any output of the v1.1 14-entry stress map, or a protected
  word, so the stress test stays unseen.
"""
import random

from .common import STRESS_MAP, STRESS_OUTPUTS

PROTECTED = frozenset({
    'nahi', 'nhi', 'nahin', 'nai', 'nh', 'na', 'naa', 'mat', 'mt', 'not', 'no', 'nope', 'never', 'dont',
    'bina', 'without', 'koi', 'kabhi', 'nothing', 'none', 'cant', 'wont', 'didnt', 'isnt', 'doesnt',
})
VOWELS = 'aeiou'
QWERTY_NEIGHBOURS = {
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl',
    'p': 'ol', 'a': 'qsz', 's': 'awdz', 'd': 'sefx', 'f': 'drgc', 'g': 'fthv', 'h': 'gyjb', 'j': 'hukn',
    'k': 'jilm', 'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}
FORBIDDEN = STRESS_OUTPUTS | PROTECTED | frozenset(STRESS_MAP)


def vowel_drop(t, rng):
    inner = [i for i in range(1, len(t) - 1) if t[i] in VOWELS]
    if not inner:
        return t
    i = rng.choice(inner)
    return t[:i] + t[i + 1:]


def keyboard_swap(t, rng):
    i = rng.randrange(len(t))
    nb = QWERTY_NEIGHBOURS.get(t[i])
    return t if not nb else t[:i] + rng.choice(nb) + t[i + 1:]


def elongate(t, rng):
    vs = [i for i, c in enumerate(t) if c in VOWELS]
    if not vs:
        return t
    i = vs[-1]
    return t[:i + 1] + t[i] * rng.choice((2, 3)) + t[i + 1:]   # at least 3 copies: "accha" -> "acchaaa"


def transpose(t, rng):
    if len(t) < 4:
        return t
    i = rng.randrange(1, len(t) - 2)
    return t[:i] + t[i + 1] + t[i] + t[i + 2:]


OPS = (vowel_drop, keyboard_swap, elongate, transpose)


def editable(tok):
    return len(tok) >= 3 and tok.isascii() and tok.isalpha() and tok.lower() not in PROTECTED


def char_noise(text, rng, p=0.25):
    """Apply at most one edit to each editable token with probability p."""
    out = []
    for tok in text.split(' '):
        if editable(tok) and rng.random() < p:
            new = rng.choice(OPS)(tok, rng)
            if new.lower() in FORBIDDEN or not new:
                new = tok
            out.append(new)
        else:
            out.append(tok)
    return ' '.join(out)


def noisy_variant(text, seed, p=0.25):
    return char_noise(text, random.Random(seed), p)
