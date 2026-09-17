"""Khmer Word Segmenter using Trie Longest Matching (MaxMatch) with Syllable Fallback and 37k RAC Dictionary."""

from pathlib import Path
from typing import List, Set, Optional
from ..normalizer.syllable_parser import split_into_syllables
from ..normalizer import normalize_khmer_text

# Path to official bundled RAC dictionary
BUNDLED_DICT_PATH = Path(__file__).parent / "dictionary" / "khmer_words.txt"

# Core vocabulary of frequently used Khmer words (administration, daily life, geography, culture, technology)
DEFAULT_KHMER_DICTIONARY = [
    # Government & Nation
    "ព្រះរាជាណាចក្រ", "កម្ពុជា", "ជាតិ", "សាសនា", "ព្រះមហាក្សត្រ",
    "រាជធានី", "ភ្នំពេញ", "ខេត្ត", "ស្រុក", "ខណ្ឌ", "ឃុំ", "សង្កាត់", "ភូមិ",
    "ក្រសួង", "មន្ទីរ", "រដ្ឋបាល", "រដ្ឋាភិបាល", "រាជរដ្ឋាភិបាល", "សាលា", "តុលាការ",
    "ច្បាប់", "សេចក្តីសម្រេច", "អនុក្រឹត្យ", "ព្រះរាជក្រឹត្យ", "ប្រកាស", "លិខិត",
    "អត្តសញ្ញាណប័ណ្ណ", "សញ្ជាតិ", "ខ្មែរ", "លិខិតឆ្លងដែន", "វិញ្ញាបនបត្រ", "ប័ណ្ណ",
    # Society & Culture
    "ប្រជាពលរដ្ឋ", "ប្រជាជន", "សង្គម", "វប្បធម៌", "ប្រពៃណី", "ទំនៀមទម្លាប់",
    "ប្រាសាទ", "អង្គរវត្ត", "បាយ័ន", "បេតិកភណ្ឌ", "ពិភពលោក", "ប្រវត្តិសាស្ត្រ",
    "សន្តិភាព", "ស្ថិរភាព", "ឯករាជ្យ", "អធិបតេយ្យ", "បូរណភាពទឹកដី",
    # Education & Science
    "សាកលវិទ្យាល័យ", "វិទ្យាល័យ", "អនុវិទ្យាល័យ", "បឋមសិក្សា", "អប់រំ", "យុវជន", "កីឡា",
    "វិទ្យាសាស្ត្រ", "បច្ចេកវិទ្យា", "នវានុវត្តន៍", "ព័ត៌មានវិទ្យា", "ទូរគមនាគមន៍",
    "បញ្ញាសិប្បនិម្មិត", "ប្រព័ន្ធ", "ទិន្នន័យ", "ស្វ័យប្រវត្ត", "ឌីជីថល",
    # Economy & Health
    "សេដ្ឋកិច្ច", "ពាណិជ្ជកម្ម", "ឧស្សាហកម្ម", "កសិកម្ម", "ទេសចរណ៍", "ហិរញ្ញវត្ថុ",
    "ធនាគារ", "ប្រាក់", "រៀល", "ដុល្លារ", "តម្លៃ", "របាយការណ៍", "កិច្ចព្រមព្រៀង",
    "សុខភាព", "សុខុមាលភាព", "មន្ទីរពេទ្យ", "វេជ្ជសាស្ត្រ", "ឱសថ", "ការងារ",
    # Nature & Geography
    "ទន្លេ", "មេគង្គ", "ទន្លេសាប", "បឹង", "សមុទ្រ", "ភ្នំ", "ព្រៃឈើ", "បរិស្ថាន",
    "ធនធានធម្មជាតិ", "ការការពារ", "អភិវឌ្ឍន៍", "ចីរភាព", "សៀមរាប", "បាត់ដំបង", "កំពត",
    # Common functional words, verbs, adjectives
    "និង", "នៃ", "ក្នុង", "លើ", "ក្រោម", "ដោយ", "ដើម្បី", "ជាមួយ", "សម្រាប់", "អំពី",
    "ជា", "មាន", "បាន", "ធ្វើ", "ទៅ", "មក", "នៅ", "ឱ្យ", "ឲ្យ", "ឃើញ", "ដឹង", "រៀន", "រស់នៅ",
    "ថ្មី", "ចាស់", "ធំ", "តូច", "ល្អ", "ច្រើន", "តិច", "ទាំងអស់", "ផ្សេងៗ",
    "ថ្ងៃ", "ខែ", "ឆ្នាំ", "វេលា", "ម៉ោង", "នាទី", "ពេល", "វេលា",
    "សួស្តី", "សូម", "អរគុណ", "គោរព", "អញ្ជើញ", "ចូលរួម", "ផ្លូវការ",
    "ខ្ញុំ", "អ្នក", "គាត់", "យើង", "ពួកគេ", "ឯង", "ចាស់"
]


class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class KhmerWordSegmenter:
    """Tokenizes continuous Khmer text into words using Trie Longest Matching."""

    def __init__(self, custom_dictionary: List[str] | None = None, load_bundled_dict: bool = True):
        self.root = TrieNode()
        self.words_count = 0

        # 1. Load official RAC dictionary if available
        if load_bundled_dict and BUNDLED_DICT_PATH.exists():
            self.load_wordlist_file(str(BUNDLED_DICT_PATH))

        # 2. Add default core vocabulary
        for w in DEFAULT_KHMER_DICTIONARY:
            self.add_word(w)

        # 3. Add custom dictionary if provided
        if custom_dictionary:
            for w in custom_dictionary:
                self.add_word(w)

    def add_word(self, word: str) -> None:
        """Adds a word into the Trie."""
        word = normalize_khmer_text(word, strip_zwsp=True)
        if not word:
            return
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True
        self.words_count += 1

    def load_wordlist_file(self, filepath: str) -> int:
        """Loads words from a newline-separated text file."""
        count = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip()
                if w:
                    self.add_word(w)
                    count += 1
        return count

    def segment(self, text: str) -> List[str]:
        """Segments continuous Khmer text into a list of words.
        Uses Dynamic Programming (DP) with Trie dictionary matching and
        orphan cluster prevention (e.g. preserves របស់, រស់, ចាស់, ទេសចរណ៍).
        """
        text = normalize_khmer_text(text, strip_zwsp=True)
        if not text:
            return []

        # Split into syllables first to ensure we never chop within a syllable cluster
        syllables = split_into_syllables(text)
        n = len(syllables)
        if n == 0:
            return []

        memo: dict[int, list[tuple[int, str, float]]] = {}

        def get_matches(i: int) -> list[tuple[int, str, float]]:
            if i in memo:
                return memo[i]
            matches: list[tuple[int, str, float]] = []

            # Non-Khmer or whitespace/punctuation forms an atomic token
            s0 = syllables[i]
            if s0.isspace() or not any(0x1780 <= ord(c) <= 0x17FF for c in s0):
                matches.append((i + 1, s0, 0.0))
                memo[i] = matches
                return matches

            prefix = ""
            for j in range(i, min(n, i + 12)):
                s = syllables[j]
                if s.isspace() or not any(0x1780 <= ord(c) <= 0x17FF for c in s):
                    break
                prefix += s

                matched_in_trie = True
                temp_node = self.root
                for ch in prefix:
                    if ch in temp_node.children:
                        temp_node = temp_node.children[ch]
                    else:
                        matched_in_trie = False
                        break

                if matched_in_trie and temp_node.is_word:
                    # Valid dictionary word: lowest cost (prefer slightly longer words)
                    cost = 1.0 - 0.001 * len(prefix)
                    matches.append((j + 1, prefix, cost))

            # Fallback single syllable
            # Heavy penalty if single syllable is an impossible orphan like ស់, ល់, ៗ
            # or starts with a dependent vowel/diacritic
            is_orphan = (
                s0 in {"ស់", "ល់", "ៗ"}
                or (len(s0) > 1 and s0[1] in "\u17cb\u17cd\u17cf\u17d0\u17d2")
                or (len(s0) > 0 and 0x17B4 <= ord(s0[0]) <= 0x17D3)
            )
            syl_cost = 100.0 if is_orphan else 5.0
            matches.append((i + 1, s0, syl_cost))

            memo[i] = matches
            return matches

        # Backward DP
        dp = [float("inf")] * (n + 1)
        dp[n] = 0.0
        parent: list[tuple[int, str] | None] = [None] * (n + 1)

        for i in range(n - 1, -1, -1):
            best_cost = float("inf")
            best_trans = None
            for next_i, word, cost in get_matches(i):
                total_cost = cost + dp[next_i]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_trans = (next_i, word)
            dp[i] = best_cost
            parent[i] = best_trans

        # Reconstruct optimal segmentation
        words: List[str] = []
        curr_i = 0
        while curr_i < n:
            trans = parent[curr_i]
            if trans is None:
                words.append(syllables[curr_i])
                curr_i += 1
            else:
                next_i, word = trans
                words.append(word)
                curr_i = next_i

        return words

    def segment_to_string(self, text: str, delimiter: str = " ") -> str:
        """Convenience method: returns word-delimited string."""
        words = self.segment(text)
        # Filter out multiple consecutive spaces
        clean_words = [w for w in words if w.strip()]
        return delimiter.join(clean_words)


_DEFAULT_SEGMENTER: Optional[KhmerWordSegmenter] = None


def get_khmer_word_segmenter() -> KhmerWordSegmenter:
    """Returns singleton instance of KhmerWordSegmenter with 37k RAC dictionary loaded."""
    global _DEFAULT_SEGMENTER
    if _DEFAULT_SEGMENTER is None:
        _DEFAULT_SEGMENTER = KhmerWordSegmenter()
    return _DEFAULT_SEGMENTER


def chunk_text_by_words(
    text: str,
    segmenter: Optional[KhmerWordSegmenter] = None,
    min_chars: int = 24,
    max_chars: int = 45
) -> List[str]:
    """Chunks continuous Khmer text into line-sized snippets strictly at whole-word boundaries.

    Guarantees that words are NEVER broken across line breaks (e.g. preserves 'ចាស់',
    'ទេសចរណ៍', 'អាស្រ័យ', 'ទេវបដិមា' completely intact).
    """
    if not text or not text.strip():
        return []

    seg = segmenter or get_khmer_word_segmenter()
    words = seg.segment(text.strip())

    lines: List[str] = []
    curr_tokens: List[str] = []
    curr_len = 0

    for w in words:
        w_len = len(w)

        # If adding w exceeds max_chars and current chunk has reached min_chars, flush line
        if (curr_len + w_len) > max_chars and curr_len >= min_chars:
            line_str = "".join(curr_tokens).strip()
            if line_str:
                lines.append(line_str)
            curr_tokens = [w] if not w.isspace() else []
            curr_len = w_len if not w.isspace() else 0
            continue

        # If w contains punctuation, prefer breaking naturally if min length reached
        if any(p in w for p in ["។", "៕", "?", "!"]) and curr_len >= min_chars:
            curr_tokens.append(w)
            line_str = "".join(curr_tokens).strip()
            if line_str:
                lines.append(line_str)
            curr_tokens = []
            curr_len = 0
            continue

        curr_tokens.append(w)
        curr_len += w_len

    if curr_tokens:
        rem_str = "".join(curr_tokens).strip()
        if rem_str:
            # If remaining fragment is short and can be appended to previous line cleanly
            if lines and (len(lines[-1]) + len(rem_str) + 1) <= (max_chars + 10):
                lines[-1] = f"{lines[-1]} {rem_str}".strip()
            elif len(rem_str) >= 6:
                lines.append(rem_str)

    return lines
