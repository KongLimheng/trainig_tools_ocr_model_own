"""Khmer Word Segmenter using Trie Longest Matching (MaxMatch) with Syllable Fallback."""

from typing import List, Set
from ..normalizer.syllable_parser import split_into_syllables
from ..normalizer import normalize_khmer_text

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
    "ខ្ញុំ", "អ្នក", "គាត់", "យើង", "ពួកគេ", "ឯង",
]


class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class KhmerWordSegmenter:
    """Tokenizes continuous Khmer text into words using Trie Longest Matching."""

    def __init__(self, custom_dictionary: List[str] | None = None):
        self.root = TrieNode()
        self.words_count = 0
        vocab = custom_dictionary or DEFAULT_KHMER_DICTIONARY
        for w in vocab:
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
        Uses Longest Matching (MaxMatch) with orthographic syllable fallback.
        """
        text = normalize_khmer_text(text, strip_zwsp=True)
        if not text:
            return []

        # Split into syllables first to ensure we never chop within a syllable cluster
        syllables = split_into_syllables(text)
        words: List[str] = []

        i = 0
        n = len(syllables)

        while i < n:
            # If current token is whitespace or punctuation, preserve it as a separate token
            if syllables[i].isspace() or not any(0x1780 <= ord(c) <= 0x17FF for c in syllables[i]):
                words.append(syllables[i])
                i += 1
                continue

            longest_word = None
            longest_match_idx = i

            current_prefix = ""
            node = self.root

            # Greedily search for longest word starting at syllable i
            for j in range(i, min(n, i + 8)):  # Max word length ~8 syllables
                syl = syllables[j]
                current_prefix += syl

                # Check if current_prefix exists in Trie
                # Check character by character in Trie
                matched_in_trie = True
                temp_node = self.root
                for ch in current_prefix:
                    if ch in temp_node.children:
                        temp_node = temp_node.children[ch]
                    else:
                        matched_in_trie = False
                        break

                if matched_in_trie and temp_node.is_word:
                    longest_word = current_prefix
                    longest_match_idx = j + 1

            if longest_word is not None:
                words.append(longest_word)
                i = longest_match_idx
            else:
                # Fallback: single syllable
                words.append(syllables[i])
                i += 1

        return words

    def segment_to_string(self, text: str, delimiter: str = " ") -> str:
        """Convenience method: returns word-delimited string."""
        words = self.segment(text)
        # Filter out multiple consecutive spaces
        clean_words = [w for w in words if w.strip()]
        return delimiter.join(clean_words)
