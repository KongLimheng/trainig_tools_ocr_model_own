"""Khmer Fonts Downloader, Unicode Validator & Deduplication Engine."""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

from .fonts import is_khmer_font, classify_font_style, STYLE_KHATT, STYLE_MOUL, STYLE_CHRIENG

# Standard User-Agent for downloads
USER_AGENT = "KhmerOCRStudio/1.0 (Khmer Font Downloader Suite; https://github.com/KongLimheng/trainig_tools_ocr_model_own)"

# Curated High-Quality Open-Source Khmer Font Packages
FONT_SOURCES = {
    "sbbic": {
        "name": "SBBIC All Khmer Unicode Fonts (Kingdomcam)",
        "url": "https://raw.githubusercontent.com/Kingdomcam/All-Fonts-Unicode-Khmer/main/All%20Khmer%20Unicode%20Fonts.zip",
        "description": "Comprehensive collection curated by the Society for Better Books in Cambodia (~258 fonts)",
    },
    "sbbic_alt": {
        "name": "All Khmer Fonts Alternative Pack (Kingdomcam)",
        "url": "https://raw.githubusercontent.com/Kingdomcam/All-Fonts-Unicode-Khmer/main/All-Khmer-Fonts.zip",
        "description": "Secondary comprehensive archive of Khmer fonts (~258 fonts)",
    },
    "7piseth": {
        "name": "7PiSeth Khmer Fonts Internet Collection",
        "url": "https://raw.githubusercontent.com/7PiSeth/khmer-font/master/public/fonts/khmer-fonts-collection.zip",
        "description": "Curated modern Khmer fonts from 7PiSeth collection (~185 fonts)",
    },
    "chamnan": {
        "name": "Chamnan-dev All Khmer Fonts Collection",
        "url": "https://codeload.github.com/chamnan-dev/All-khmer-fonts/zip/refs/heads/master",
        "description": "Extensive font collection with display and text fonts (~262 fonts)",
    },
    "khmeros": {
        "name": "Khmer Software Initiative (Open Forum / Danh Hong)",
        "url": "https://codeload.github.com/KhmerSoftwareInitiative/khmer-unicode-fonts/zip/refs/heads/master",
        "description": "Official national standard KhmerOS fonts (v3.1, v4.0, v5.0, ~40 fonts)",
    },
}

# Curated Google Fonts with Khmer script
GOOGLE_KHMER_FONTS = [
    "Angkor", "Battambang", "Bayon", "Bokor", "Chenla", "Content",
    "Dangrek", "Fasthand", "Freehand", "Hanuman", "Kantumruy Pro",
    "Kdam Thmor Pro", "Khmer", "Koh Santepheap", "Konkhmer Sleokchher",
    "Koulen", "Metal", "Moul", "Moulpali", "Nokora", "Noto Sans Khmer",
    "Noto Serif Khmer", "Odor Mean Chey", "Preahvihear", "Siemreap",
    "Suwannaphum", "Taprom"
]


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for exact de-duplication."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_font_postscript_name(filepath: Path) -> str:
    """Extracts the PostScript name from a font file using fontTools, or falls back to stem."""
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(str(filepath), fontNumber=0)
        name_table = tt.get("name")
        if name_table:
            for record in name_table.names:
                # Name ID 6 = PostScript name
                if record.nameID == 6:
                    return record.toUnicode()
    except Exception:
        pass
    return filepath.stem


class KhmerFontDownloader:
    """Downloads, extracts, validates Unicode compatibility, deduplicates, and catalogs Khmer fonts."""

    def __init__(
        self,
        output_dir: str | Path = "fonts",
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self.seen_hashes: set[str] = set()
        self.seen_ps_names: set[str] = set()

    def _log(self, message: str, percent: float = -1.0) -> None:
        """Sends progress update to callback and console."""
        if self.progress_callback:
            self.progress_callback(message, percent)
        else:
            print(f"[KhmerFontDownloader] {message}")

    def download_file(self, url: str, dest_path: Path) -> bool:
        """Downloads a remote file with progress tracking."""
        self._log(f"Downloading from: {url}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                block_size = 65536

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        if int(pct) % 20 == 0:
                            self._log(f"  Downloaded {downloaded // 1024} KB / {total_size // 1024} KB ({pct:.1f}%)", pct)

            self._log(f"Download complete: {dest_path.name} ({dest_path.stat().st_size // 1024} KB)")
            return True
        except Exception as e:
            self._log(f"Error downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    def ingest_zip(self, zip_path: Path, temp_extract_dir: Path) -> List[Path]:
        """Extracts all font files (.ttf, .otf, .ttc) from a ZIP archive."""
        extracted: List[Path] = []
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.namelist():
                    lower_name = member.lower()
                    if lower_name.endswith((".ttf", ".otf", ".ttc")):
                        filename = Path(member).name
                        if not filename or filename.startswith("."):
                            continue
                        target = temp_extract_dir / filename
                        # Handle collision inside the same zip
                        counter = 1
                        while target.exists():
                            target = temp_extract_dir / f"{Path(member).stem}_{counter}{Path(member).suffix}"
                            counter += 1

                        with z.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted.append(target)
        except Exception as e:
            self._log(f"Failed to extract ZIP {zip_path}: {e}")
        return extracted

    def ingest_directory(self, source_dir: Path) -> List[Path]:
        """Collects all font files from an existing directory."""
        fonts: List[Path] = []
        for ext in (".ttf", ".otf", ".ttc"):
            fonts.extend(list(source_dir.rglob(f"*{ext}")))
            fonts.extend(list(source_dir.rglob(f"*{ext.upper()}")))
        return fonts

    def process_and_install_fonts(self, candidate_fonts: List[Path]) -> Dict[str, any]:
        """Audits candidate fonts: validates Unicode Khmer support, deduplicates, and installs to output_dir."""
        installed = 0
        skipped_non_khmer = 0
        skipped_duplicate = 0
        installed_fonts_info = []

        # Index existing fonts in output_dir first
        for existing in self.output_dir.glob("*.[to]tf"):
            try:
                h = compute_file_sha256(existing)
                self.seen_hashes.add(h)
                ps = get_font_postscript_name(existing).strip().lower()
                if ps:
                    self.seen_ps_names.add(ps)
            except Exception:
                pass

        self._log(f"Evaluating {len(candidate_fonts)} candidate font files...")

        for font_file in candidate_fonts:
            if not font_file.exists() or font_file.stat().st_size == 0:
                continue

            # 1. Unicode Khmer validation (must have codepoints in U+1780 - U+17FF)
            if not is_khmer_font(font_file):
                skipped_non_khmer += 1
                continue

            # 2. Exact file hash deduplication
            file_hash = compute_file_sha256(font_file)
            if file_hash in self.seen_hashes:
                skipped_duplicate += 1
                continue

            # 3. PostScript font name deduplication (avoids renames of identical font revisions)
            ps_name = get_font_postscript_name(font_file).strip().lower()
            if ps_name and ps_name in self.seen_ps_names:
                skipped_duplicate += 1
                continue

            # Clean destination filename
            clean_name = font_file.name.replace(" ", "_")
            dest_file = self.output_dir / clean_name
            counter = 1
            while dest_file.exists():
                dest_file = self.output_dir / f"{font_file.stem}_{counter}{font_file.suffix}"
                counter += 1

            shutil.copy2(font_file, dest_file)
            self.seen_hashes.add(file_hash)
            if ps_name:
                self.seen_ps_names.add(ps_name)

            installed += 1
            style = classify_font_style(dest_file.stem)
            installed_fonts_info.append({
                "filename": dest_file.name,
                "style": style,
                "size_kb": dest_file.stat().st_size // 1024,
                "postscript_name": ps_name or dest_file.stem,
            })

        self._log(f"Processed: +{installed} installed, {skipped_duplicate} duplicates skipped, {skipped_non_khmer} non-Khmer/legacy filtered out.")
        return {
            "installed": installed,
            "skipped_duplicate": skipped_duplicate,
            "skipped_non_khmer": skipped_non_khmer,
            "fonts": installed_fonts_info,
        }

    def generate_catalog(self) -> Path:
        """Generates an inventory catalog (fonts_catalog.json) of all fonts in output_dir."""
        catalog_path = self.output_dir / "fonts_catalog.json"
        font_files = list(self.output_dir.glob("*.[to]tf"))

        by_style = {STYLE_KHATT: [], STYLE_MOUL: [], STYLE_CHRIENG: []}
        all_catalog = []

        for p in font_files:
            style = classify_font_style(p.stem)
            entry = {
                "name": p.stem,
                "filename": p.name,
                "style": style,
                "size_bytes": p.stat().st_size,
            }
            all_catalog.append(entry)
            by_style.setdefault(style, []).append(p.stem)

        summary = {
            "total_fonts": len(font_files),
            "styles_breakdown": {
                STYLE_KHATT: len(by_style.get(STYLE_KHATT, [])),
                STYLE_MOUL: len(by_style.get(STYLE_MOUL, [])),
                STYLE_CHRIENG: len(by_style.get(STYLE_CHRIENG, [])),
            },
            "fonts": all_catalog,
        }

        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._log(f"Catalog saved to {catalog_path}: {len(font_files)} total Khmer fonts indexed.")
        return catalog_path

    def run_download(
        self,
        source: str = "all",
        custom_url: Optional[str] = None,
        custom_zip: Optional[str | Path] = None,
        custom_folder: Optional[str | Path] = None,
        clean: bool = False
    ) -> Dict[str, any]:
        """Executes full download and ingestion workflow."""
        if clean and self.output_dir.exists():
            self._log(f"Cleaning output directory: {self.output_dir}")
            for item in self.output_dir.iterdir():
                if item.name == "README.md":
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        temp_workspace = self.output_dir / ".tmp_downloader"
        if temp_workspace.exists():
            shutil.rmtree(temp_workspace)
        temp_workspace.mkdir(parents=True, exist_ok=True)

        candidate_fonts: List[Path] = []

        try:
            # 1. Custom folder ingestion
            if custom_folder:
                folder_path = Path(custom_folder)
                if folder_path.exists():
                    self._log(f"Ingesting from local directory: {folder_path}")
                    candidate_fonts.extend(self.ingest_directory(folder_path))

            # 2. Custom zip ingestion
            if custom_zip:
                zip_p = Path(custom_zip)
                if zip_p.exists():
                    self._log(f"Ingesting from local ZIP file: {zip_p}")
                    extract_sub = temp_workspace / "custom_zip"
                    extract_sub.mkdir(exist_ok=True)
                    candidate_fonts.extend(self.ingest_zip(zip_p, extract_sub))

            # 3. Custom URL download
            if custom_url:
                target_zip = temp_workspace / "custom_url_download.zip"
                if self.download_file(custom_url, target_zip):
                    extract_sub = temp_workspace / "custom_url"
                    extract_sub.mkdir(exist_ok=True)
                    candidate_fonts.extend(self.ingest_zip(target_zip, extract_sub))

            # 4. Built-in curated source packages
            sources_to_fetch = []
            if source == "all":
                sources_to_fetch = list(FONT_SOURCES.keys())
            elif source in FONT_SOURCES:
                sources_to_fetch = [source]

            for s_key in sources_to_fetch:
                s_info = FONT_SOURCES[s_key]
                self._log(f"\n--- Fetching {s_info['name']} ---")
                pack_zip = temp_workspace / f"{s_key}.zip"
                if self.download_file(s_info["url"], pack_zip):
                    extract_sub = temp_workspace / s_key
                    extract_sub.mkdir(exist_ok=True)
                    candidate_fonts.extend(self.ingest_zip(pack_zip, extract_sub))

            # Process, validate Unicode, deduplicate, and install
            results = self.process_and_install_fonts(candidate_fonts)

            # Generate inventory catalog
            catalog_file = self.generate_catalog()
            results["catalog_file"] = str(catalog_file)
            results["total_fonts_in_dir"] = len(list(self.output_dir.glob("*.[to]tf")))

            return results

        finally:
            if temp_workspace.exists():
                shutil.rmtree(temp_workspace, ignore_errors=True)
