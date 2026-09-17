"""Character Error Rate (CER) and Exact Match Metrics."""

from typing import Sequence


def levenshtein_distance(ref: Sequence, hyp: Sequence) -> int:
    """Computes Levenshtein edit distance between two sequences."""
    try:
        import editdistance
        return editdistance.eval(ref, hyp)
    except ImportError:
        # Pure Python fallback
        r_len = len(ref)
        h_len = len(hyp)
        dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

        for i in range(r_len + 1):
            dp[i][0] = i
        for j in range(h_len + 1):
            dp[0][j] = j

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[r_len][h_len]


def calculate_cer(targets: list[str], predictions: list[str]) -> float:
    """Calculates standard Character Error Rate: sum(dist) / sum(target_len).
    Returns value between 0.0 (perfect) and 1.0+ (worse).
    """
    total_dist = 0
    total_len = 0

    for target, pred in zip(targets, predictions):
        dist = levenshtein_distance(target, pred)
        total_dist += dist
        total_len += len(target)

    return total_dist / max(1, total_len)


def calculate_exact_match(targets: list[str], predictions: list[str]) -> float:
    """Calculates the ratio of exactly matched sequences (0.0 to 1.0)."""
    if not targets:
        return 0.0
    matches = sum(1 for t, p in zip(targets, predictions) if t == p)
    return matches / len(targets)
