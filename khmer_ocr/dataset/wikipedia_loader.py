"""Khmer Wikipedia Data Harvester & Corpus Pipeline (km.wikipedia.org)."""

import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from ..normalizer import normalize_khmer_text
from ..normalizer.syllable_parser import split_into_syllables
from ..postprocess.word_segmenter import chunk_text_by_words

WIKIPEDIA_API_URL = "https://km.wikipedia.org/w/api.php"
USER_AGENT = "KhmerOCRStudio/1.0 (AI Khmer OCR Training Suite; contact@khmer-ocr.org)"

# High-density curated articles across rich Khmer domains
CURATED_TOPICS = {
    "History & Angkor": [
        "ព្រះរាជាណាចក្រកម្ពុជា",
        "ប្រវត្តិសាស្ត្រខ្មែរ",
        "អង្គរវត្ត",
        "អាណាចក្រខ្មែរ",
        "ប្រាសាទបាយ័ន",
        "សម័យអង្គរ",
        "ព្រះបាទជ័យវរ្ម័នទី៧",
    ],
    "Geography & Nature": [
        "ភូមិសាស្ត្រកម្ពុជា",
        "រាជធានីភ្នំពេញ",
        "ទន្លេមេគង្គ",
        "បឹងទន្លេសាប",
        "ខេត្តសៀមរាប",
        "ខេត្តបាត់ដំបង",
        "ខេត្តកំពត",
        "ភ្នំក្រវាញ",
    ],
    "Culture & Literature": [
        "វប្បធម៌ខ្មែរ",
        "អក្សរសាស្ត្រខ្មែរ",
        "របាំព្រះរាជទ្រព្យ",
        "ភាសាខ្មែរ",
        "បុណ្យអុំទូក",
        "ពិធីបុណ្យចូលឆ្នាំខ្មែរ",
        "រឿងរាមកេរ្តិ៍",
    ],
    "Law, Economy & Governance": [
        "រដ្ឋធម្មនុញ្ញនៃព្រះរាជាណាចក្រកម្ពុជា",
        "សេដ្ឋកិច្ចកម្ពុជា",
        "រដ្ឋសភាកម្ពុជា",
        "ធនាគារជាតិនៃកម្ពុជា",
        "ប្រាក់រៀល",
        "កសិកម្មនៅកម្ពុជា",
    ],
    "Science & Technology": [
        "វិទ្យាសាស្ត្រ",
        "បច្ចេកវិទ្យា",
        "បញ្ញាសិប្បនិម្មិត",
        "ព័ត៌មានវិទ្យា",
        "អប់រំនៅកម្ពុជា",
        "សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ",
    ],
}


# Internal maintenance and meta namespaces on km.wikipedia.org to exclude
IGNORED_NAMESPACES = (
    "វិគីភិឌា:", "ជំនួយ:", "ទំព័រគំរូ:", "ឯកសារ:", "ចាត់ថ្នាក់:",
    "User:", "File:", "Wikipedia:", "Template:", "Portal:", "Special:", "MediaWiki:", "Category:"
)


class KhmerWikipediaLoader:
    """Streams, cleans, and slices authentic text from Khmer Wikipedia (km.wikipedia.org)."""

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

    def _api_get(self, params: Dict[str, str]) -> dict:
        """Helper to send MediaWiki API query."""
        params["format"] = "json"
        query_string = urllib.parse.urlencode(params)
        url = f"{WIKIPEDIA_API_URL}?{query_string}"
        query_str = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{WIKIPEDIA_API_URL}?{query_str}",
            headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_article_extract(self, title: str) -> str:
        """Fetches plain text extract of a single article."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "redirects": "1",
        }
        data = self._api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for _, page_info in pages.items():
            if "extract" in page_info:
                return page_info["extract"]
        return ""

    def search_articles(self, query: str, limit: int = 15) -> List[str]:
        """Searches for article titles matching a keyword, excluding maintenance namespaces."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(min(limit * 2, 50)),
        }
        data = self._api_get(params)
        items = data.get("query", {}).get("search", [])
        titles = []
        for item in items:
            title = item.get("title", "")
            if title and not any(title.startswith(prefix) for prefix in IGNORED_NAMESPACES):
                titles.append(title)
                if len(titles) >= limit:
                    break
        return titles

    def fetch_random_titles(self, count: int = 15) -> List[str]:
        """Fetches random article titles, excluding maintenance namespaces."""
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": str(min(count * 2, 50)),
        }
        data = self._api_get(params)
        items = data.get("query", {}).get("random", [])
        titles = []
        for item in items:
            title = item.get("title", "")
            if title and not any(title.startswith(prefix) for prefix in IGNORED_NAMESPACES):
                titles.append(title)
                if len(titles) >= count:
                    break
        return titles

    def clean_and_segment_extract(
        self,
        extract: str,
        pure_khmer: bool = True,
        min_khmer_ratio: float = 0.70,
        allow_latin: bool = False,
    ) -> List[str]:
        """Cleans wikitext extract and segments into authentic Khmer sentence lines.

        Eliminates parenthetical foreign glosses, non-Khmer scripts (Chinese, Thai, Lao, Cyrillic, etc.),
        and enforces authentic pure Khmer script lines (0 Latin/English letters by default).
        """
        if not extract:
            return []

        # 1. Remove section headers (e.g. == ប្រវត្តិសាស្ត្រ ==)
        cleaned = re.sub(r"==+[^=]+==+", "\n", extract)

        # 2. Remove IPA pronunciation guides (e.g. [kɑmˈpuˈciə])
        cleaned = re.sub(r"\[[a-zA-Z0-9\sˈˌːˑɪʊəɛɔæɑθðʃʒŋʔʰʲʷ'\"/.,\-~]+\]", " ", cleaned)

        # 3. Remove citations and references (e.g. [1], [ក], [១២])
        cleaned = re.sub(r"\[[0-9\u17E0-\u17E9a-zA-Z\u1780-\u17B3]+\]", " ", cleaned)

        # 4. Remove URL links and external formatting
        cleaned = re.sub(r"https?://\S+", " ", cleaned)

        # 5. Strip parenthetical foreign language glosses
        # e.g. ( ចិន: 土地神屋 អង់គ្លេស: Spirit houses ថៃ: ศាលพระภูมิ )
        cleaned = re.sub(
            r"\([^)]*(?:ចិន|ថៃ|អង់គ្លេស|បារាំង|អាល្លឺម៉ង់|រុស្ស៊ី|ឡាតាំង|English|French|Thai|Chinese|German|Latin|ភាសា|ភិងអ៊ិង|Zhōng|[\u4e00-\u9fff\u0e00-\u0eff])[^)]*\)",
            " ",
            cleaned
        )
        # Strip pure Latin / numeric acronyms inside parentheses e.g. (GDP), (National Assembly), (16th century)
        cleaned = re.sub(r"\([a-zA-Z0-9\s,.:;\-/'\"]+\)", " ", cleaned)

        # 6. Strip non-Khmer foreign scripts (Chinese/CJK, Japanese, Korean, Thai, Lao, Cyrillic, Greek, Arabic, Devanagari, Latin diacritics/pinyin)
        cleaned = re.sub(
            r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0eff\u0400-\u04ff\u0370-\u03ff\u0600-\u06ff\u0900-\u097f\u00c0-\u024f\u1e00-\u1eff]",
            " ",
            cleaned
        )

        # 7. Remove residual empty parentheses or brackets
        cleaned = re.sub(r"\(\s*\)", " ", cleaned)
        cleaned = re.sub(r"\[\s*\]", " ", cleaned)

        # 8. Segment into paragraphs using newlines
        paragraphs = re.split(r"[\n\r]+", cleaned)

        sentences = []
        seen = set()

        for para in paragraphs:
            # Normalize Khmer Unicode canonically & remove ZWSP
            norm = normalize_khmer_text(para.strip(), strip_zwsp=True)
            if not norm:
                continue

            # Remove multiple spaces
            norm = re.sub(r"\s+", " ", norm).strip()

            # Filter: must contain Khmer characters
            khmer_chars = [c for c in norm if 0x1780 <= ord(c) <= 0x17FF]
            if len(khmer_chars) < 10:
                continue

            # Chunk paragraph cleanly using word-level boundaries (target 24-45 chars per line)
            # Guarantees words are never split mid-syllable or mid-word (preserves words like ចាស់, ទេសចរណ៍).
            chunked = chunk_text_by_words(norm, min_chars=24, max_chars=45)
            for line_str in chunked:
                line_str = line_str.strip()
                if len(line_str) < 15:
                    continue

                # Pure Khmer check: reject any Latin characters if allow_latin is False
                if pure_khmer and not allow_latin:
                    if any("a" <= ch.lower() <= "z" for ch in line_str):
                        continue

                # Filter: density of Khmer characters
                line_khmer_chars = [c for c in line_str if 0x1780 <= ord(c) <= 0x17FF]
                if (len(line_khmer_chars) / len(line_str)) < min_khmer_ratio:
                    continue

                # Avoid orphan starting characters
                if 0x17B4 <= ord(line_str[0]) <= 0x17D3 or line_str[0] == "\u17D7":
                    continue
                if line_str.startswith("ស់") or line_str.startswith("ល់"):
                    continue

                if line_str not in seen:
                    seen.add(line_str)
                    sentences.append(line_str)

        return sentences

    def harvest_from_topics(
        self,
        categories: Optional[List[str]] = None,
        max_articles: int = 20,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pure_khmer: bool = True,
        min_khmer_ratio: float = 0.70,
        allow_latin: bool = False,
    ) -> List[str]:
        """Harvests sentences from curated Khmer topics."""
        titles_to_fetch = []
        cats = categories or list(CURATED_TOPICS.keys())
        for c in cats:
            if c in CURATED_TOPICS:
                titles_to_fetch.extend(CURATED_TOPICS[c])

        # Limit to max_articles
        titles_to_fetch = titles_to_fetch[:max_articles]
        total = len(titles_to_fetch)

        all_sentences = []
        for idx, title in enumerate(titles_to_fetch):
            if progress_callback:
                progress_callback(idx + 1, total, f"Fetching: {title}")
            try:
                extract = self.fetch_article_extract(title)
                sentences = self.clean_and_segment_extract(
                    extract,
                    pure_khmer=pure_khmer,
                    min_khmer_ratio=min_khmer_ratio,
                    allow_latin=allow_latin,
                )
                all_sentences.extend(sentences)
            except Exception:
                continue

        return all_sentences

    def harvest_from_random(
        self,
        count: int = 20,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pure_khmer: bool = True,
        min_khmer_ratio: float = 0.70,
        allow_latin: bool = False,
    ) -> List[str]:
        """Harvests sentences by crawling random articles."""
        titles = self.fetch_random_titles(count)
        total = len(titles)

        all_sentences = []
        for idx, title in enumerate(titles):
            if progress_callback:
                progress_callback(idx + 1, total, f"Crawling: {title}")
            try:
                extract = self.fetch_article_extract(title)
                sentences = self.clean_and_segment_extract(
                    extract,
                    pure_khmer=pure_khmer,
                    min_khmer_ratio=min_khmer_ratio,
                    allow_latin=allow_latin,
                )
                all_sentences.extend(sentences)
            except Exception:
                continue

        return all_sentences

    def harvest_from_search(
        self,
        query: str,
        max_articles: int = 15,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pure_khmer: bool = True,
        min_khmer_ratio: float = 0.70,
        allow_latin: bool = False,
    ) -> List[str]:
        """Searches for query on Khmer Wikipedia and extracts all matching sentences."""
        titles = self.search_articles(query, limit=max_articles)
        total = len(titles)

        all_sentences = []
        for idx, title in enumerate(titles):
            if progress_callback:
                progress_callback(idx + 1, total, f"Searching: {title}")
            try:
                extract = self.fetch_article_extract(title)
                sentences = self.clean_and_segment_extract(
                    extract,
                    pure_khmer=pure_khmer,
                    min_khmer_ratio=min_khmer_ratio,
                    allow_latin=allow_latin,
                )
                all_sentences.extend(sentences)
            except Exception:
                continue

        return all_sentences

    def save_corpus(self, sentences: List[str], output_path: str | Path) -> int:
        """Saves sentences into a newline-separated text file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    f.write(f"{s_clean}\n")
        return len(sentences)
