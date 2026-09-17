"""Khmer Character Map & Tokenizer for OCR Models."""

import json
from pathlib import Path


DEFAULT_KHMER_CHARACTERS = [
    # 33 Consonants
    "ក", "ខ", "គ", "ឃ", "ង",
    "ច", "ឆ", "ជ", "ឈ", "ញ",
    "ដ", "ឋ", "ឌ", "ឍ", "ណ",
    "ត", "ថ", "ទ", "ធ", "ន",
    "ប", "ផ", "ព", "ភ", "ម",
    "យ", "រ", "ល", "វ", "ស",
    "ហ", "ឡ", "អ",
    # Coeng (Subscript marker)
    "្",
    # Dependent Vowels
    "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ",
    "ើ", "ឿ", "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ",
    # Independent Vowels
    "ឥ", "ឦ", "ឧ", "ឩ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ", "ឯ", "ឰ", "ឱ", "ឳ",
    # Consonant Shifters & Diacritics
    "៉", "៊", "ំ", "ះ", "ៈ", "់", "៌", "៍", "៎", "៏", "័", "៑", "៓",
    # Khmer Digits
    "០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩",
    # Khmer Punctuation & Symbols
    "។", "៕", "ៗ", "៘", "៙", "៚", "៛",
    # Latin Digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Common Punctuation & Arithmetic
    " ", ".", ",", ":", ";", "!", "?", "-", "_", "(", ")", "[", "]",
    "/", "\\", "\"", "'", "+", "=", "%", "$", "@", "&", "#", "*",
    # Latin Alphabet (Uppercase & Lowercase for bilingual documents)
    *"abcdefghijklmnopqrstuvwxyz",
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
]


class KhmerCharMap:
    """Manages character-to-index and index-to-character mapping for CTC and attention models."""

    BLANK_TOKEN = "<blank>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, characters: list[str] | None = None, include_specials: bool = True):
        self.include_specials = include_specials
        self.char_list: list[str] = []

        if include_specials:
            # Special tokens: Blank is index 0 (standard for PyTorch CTCLoss)
            self.char_list = [self.BLANK_TOKEN, self.UNK_TOKEN]

        chars = characters if characters is not None else DEFAULT_KHMER_CHARACTERS
        # Keep unique characters while preserving order
        seen = set(self.char_list)
        for c in chars:
            if c not in seen:
                seen.add(c)
                self.char_list.append(c)

        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_list)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_list)}

        self.blank_idx = self.char_to_idx.get(self.BLANK_TOKEN, 0)
        self.unk_idx = self.char_to_idx.get(self.UNK_TOKEN, 1 if include_specials else 0)

    def __len__(self) -> int:
        return len(self.char_list)

    @property
    def num_classes(self) -> int:
        return len(self.char_list)

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of integer token indices."""
        return [self.char_to_idx.get(char, self.unk_idx) for char in text]

    def decode(self, indices: list[int], remove_blank: bool = True) -> str:
        """Decodes a sequence of indices back to text string."""
        chars = []
        for idx in indices:
            if remove_blank and idx == self.blank_idx:
                continue
            char = self.idx_to_char.get(idx, "")
            if char in (self.BLANK_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                continue
            chars.append(char)
        return "".join(chars)

    def decode_ctc(self, sequence: list[int]) -> str:
        """Greedy CTC decoder: collapses consecutive duplicate indices and removes blank tokens."""
        collapsed = []
        prev_idx = -1
        for idx in sequence:
            if idx != prev_idx:
                if idx != self.blank_idx:
                    char = self.idx_to_char.get(idx, "")
                    if char and char not in (self.BLANK_TOKEN, self.PAD_TOKEN, self.UNK_TOKEN):
                        collapsed.append(char)
                prev_idx = idx
        return "".join(collapsed)

    def save(self, filepath: str | Path) -> None:
        """Saves vocabulary character list to a text file (one character per line)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for char in self.char_list:
                f.write(char + "\n")

    @classmethod
    def load(cls, filepath: str | Path) -> "KhmerCharMap":
        """Loads vocabulary from a text file."""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]
        return cls(characters=lines, include_specials=False)
