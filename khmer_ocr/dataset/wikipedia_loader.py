"""Khmer Wikipedia Data Harvester & Corpus Pipeline (km.wikipedia.org)."""

import re
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from ..normalizer import normalize_khmer_text
from ..normalizer.syllable_parser import split_into_syllables

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
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data

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
        """Searches for article titles matching a keyword."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(min(limit, 50)),
        }
        data = self._api_get(params)
        items = data.get("query", {}).get("search", [])
        return [item["title"] for item in items if "title" in item]

    def fetch_random_titles(self, count: int = 15) -> List[str]:
        """Fetches random article titles."""
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": str(min(count, 50)),
        }
        data = self._api_get(params)
        items = data.get("query", {}).get("random", [])
        return [item["title"] for item in items if "title" in item]

    def clean_and_segment_extract(self, extract: str) -> List[str]:
        """Cleans wikitext extract and segments into authentic Khmer sentence lines."""
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

        # 5. Segment into raw sentence candidates using Khmer full-stop (។), (៕), ?, ! or newlines
        raw_candidates = re.split(r"[។៕\n\r]+", cleaned)

        sentences = []
        seen = set()

        for candidate in raw_candidates:
            # Normalize Khmer Unicode canonically & remove ZWSP
            norm = normalize_khmer_text(candidate.strip(), strip_zwsp=True)
            if not norm:
                continue

            # Remove multiple spaces
            norm = re.sub(r"\s+", " ", norm).strip()

            # Filter: must contain Khmer characters
            khmer_chars = [c for c in norm if 0x1780 <= ord(c) <= 0x17FF]
            if len(khmer_chars) < 10:
                continue

            # Filter: at least 60% Khmer characters to avoid foreign language lists
            if (len(khmer_chars) / max(1, len(norm))) < 0.55:
                continue

            # Chunk long sentences cleanly using orthographic syllables (target 24-45 chars per line)
            if len(norm) > 45:
                syls = split_into_syllables(norm)
                curr: list[str] = []
                curr_len = 0
                for s in syls:
                    curr.append(s)
                    curr_len += len(s)
                    # When target minimum line length is reached, break on whitespace, punctuation or max length
                    if curr_len >= 24:
                        if s.isspace() or any(p in s for p in ["។", "៕", ",", "!", "?", ";"]) or curr_len >= 42:
                            line_str = "".join(curr).strip()
                            if len(line_str) >= 12 and line_str not in seen:
                                seen.add(line_str)
                                sentences.append(line_str)
                            curr = []
                            curr_len = 0
                if curr:
                    rem = "".join(curr).strip()
                    if len(rem) >= 12 and rem not in seen:
                        seen.add(rem)
                        sentences.append(rem)
                    elif sentences and len(rem) > 0:
                        combined = f"{sentences[-1]} {rem}".strip()
                        if len(combined) <= 55:
                            sentences[-1] = combined
            else:
                if len(norm) >= 12 and norm not in seen:
                    seen.add(norm)
                    sentences.append(norm)

        return sentences

    def harvest_from_topics(
        self,
        categories: Optional[List[str]] = None,
        max_articles: int = 20,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
                sentences = self.clean_and_segment_extract(extract)
                all_sentences.extend(sentences)
            except Exception:
                continue

        return all_sentences

    def harvest_from_random(
        self,
        count: int = 20,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
                sentences = self.clean_and_segment_extract(extract)
                all_sentences.extend(sentences)
            except Exception:
                continue

        return all_sentences

    def harvest_from_search(
        self,
        query: str,
        max_articles: int = 15,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
                sentences = self.clean_and_segment_extract(extract)
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
