# 🇰🇭 Khmer Fonts Directory

This directory stores authentic Khmer TrueType (`.ttf`) and OpenType (`.otf` / `.ttc`) fonts for synthetic OCR dataset generation.

---

## 📊 Current Inventory Summary

- **Total Valid Khmer Fonts**: **418**
- **Typographic Styles Breakdown**:
  - **Standard Print (Khatt / ខាត់)**: 381 fonts
  - **Ornate / Title (Moul / មូល)**: 14 fonts
  - **Slanted / Cursive (Chrieng / ជ្រៀង)**: 23 fonts
- **Catalog File**: [`fonts_catalog.json`](file:///home/heng/Development/training_tools/fonts/fonts_catalog.json)
- **Fast Persistent Cache**: [`.font_cache.json`](file:///home/heng/Development/training_tools/fonts/.font_cache.json) (sub-300ms startup)

---

## 🚀 CLI Commands

### 1. Download & Validate All Khmer Fonts from the Internet
Downloads and extracts from all open-source repositories (SBBIC, 7PiSeth, Chamnan, KhmerOS), validates Unicode `cmap` (`U+1780` - `U+17FF`), and deduplicates:
```bash
uv run --no-sync python run_app.py download-fonts --out fonts
```

### 2. Ingest a Custom Local ZIP Archive (e.g. 1000+ fonts pack)
If you downloaded a custom ZIP file of fonts (e.g. from Google Drive or Telegram):
```bash
uv run --no-sync python run_app.py download-fonts --zip path/to/khmer_fonts_1000.zip --out fonts
```

### 3. Download from a Custom Remote URL
```bash
uv run --no-sync python run_app.py download-fonts --url https://example.com/khmer_fonts_pack.zip --out fonts
```

### 4. Synthesize OCR Training Dataset with All Downloaded Fonts
```bash
uv run --no-sync python run_app.py synth \
  --corpus data/khmer_wiki_corpus.txt \
  --fonts-dir fonts \
  --samples 35000 \
  --val-split 0.10 \
  --clean-ratio 0.40 \
  --out data/dataset_thousand_fonts
```
