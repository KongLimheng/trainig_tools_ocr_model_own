"""Khmer Glyph Confusion Matrix & Alignment Diagnostic."""

from collections import Counter
from typing import Tuple


def align_sequences(ref: str, hyp: str) -> list[Tuple[str | None, str | None]]:
    """Aligns reference and hypothesis strings using Needleman-Wunsch dynamic programming.
    Returns list of (ref_char, hyp_char) pairs. None represents an insertion/deletion.
    """
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost   # substitution
            )

    # Backtrack alignment
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))  # deletion from ref
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))  # insertion into hyp
            j -= 1

    alignment.reverse()
    return alignment


class KhmerConfusionTracker:
    """Tracks substitution errors between Khmer characters to identify confusions."""

    def __init__(self):
        self.confusion_matrix: Counter[Tuple[str, str]] = Counter()
        self.deletions: Counter[str] = Counter()
        self.insertions: Counter[str] = Counter()

    def update(self, target: str, prediction: str) -> None:
        """Updates counts from target and prediction strings."""
        alignment = align_sequences(target, prediction)
        for ref_ch, hyp_ch in alignment:
            if ref_ch is not None and hyp_ch is not None:
                if ref_ch != hyp_ch:
                    self.confusion_matrix[(ref_ch, hyp_ch)] += 1
            elif ref_ch is not None and hyp_ch is None:
                self.deletions[ref_ch] += 1
            elif ref_ch is None and hyp_ch is not None:
                self.insertions[hyp_ch] += 1

    def top_confusions(self, top_n: int = 15) -> list[dict]:
        """Returns the most frequent character substitution errors."""
        results = []
        for (ref_ch, hyp_ch), count in self.confusion_matrix.most_common(top_n):
            results.append({
                "ground_truth": ref_ch,
                "predicted": hyp_ch,
                "count": count,
            })
        return results
