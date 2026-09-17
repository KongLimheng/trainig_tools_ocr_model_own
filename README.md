# 🇰🇭 KhmerOCR Studio & Training Tools

An expert-grade AI OCR training, linguistic auditing, and deployment toolkit specifically engineered for the **Khmer language**.

Built with **PyTorch**, **PyQt5**, and managed with **`uv`**, this suite solves core orthographic challenges of Khmer digital script: multi-level stacked subscripts (Coeng `្`), vowel ligatures, continuous writing without inter-word spaces, and Unicode sequencing ambiguities.

---

## ✨ Key Features & Capabilities

### 1. 🎨 Synthetic Data Studio (HarfBuzz / LibRaqm)
- **Complex Text Shaping**: Powered by Pillow with HarfBuzz/Raqm, ensuring subscripts (`្ក`, `្ជ`, `្ឋ`, `្ញ`), shifters (`៉`, `៊`), and stacked vowels (`ើ`, `ាំ`) render without glyph clipping.
- **Font Style Filtering**: Classifies fonts by typographic style:
  - **Standard Print (Khatt / ខាត់)**: Upright formal print fonts.
  - **Ornate / Title (Moul / មូល)**: Monastic decorative header fonts.
  - **Slanted / Cursive (Chrieng / ជ្រៀង)**: Informal handwriting-like fonts.
- **Custom Corpus Importer**: Import your own `.txt` or `.csv` files to synthesize lines matching your specific domain.
- **Interactive Zoom Toolbar**: Inspect text lines at **1:1 Actual Size**, fit window, or zoom up to 400% with toggleable **Baseline & Subscript Overhang Guidelines**.

### 2. 📊 Khmer Unicode Normalizer & Dataset Auditor
- **Canonical Codepoint Reordering**: Enforces strict Unicode sequencing:
  $$\text{Base} \to \text{Shifter (៉/៊)} \to \text{Coeng (្) + Subscript} \to \text{Dependent Vowel} \to \text{Diacritics}$$
- **Zero-Width Space (`\u200B`) Sanitizer**: Eliminates invisible control codes and orphan Coeng marks from OCR training labels.
- **Multi-Format Export**: Export audited datasets to **PaddleOCR (`rec_gt.txt`)** and **CSV format** with one click.

### 3. 🚀 High-Performance Training Engine
- **Architectures**:
  - **CRNN-ResNet34-CTC**: Fast, lightweight (~15MB), optimal for local GPUs (GTX 1060 6GB) and CPU inference.
  - **CRNN-MobileNetV3-CTC**: Ultra-lightweight for mobile and edge devices.
- **Dynamic Aspect-Ratio Batching**: Eliminates wasteful zero-padding across varying line lengths, accelerating training by 2–3x.
- **Mixed Precision (AMP FP16)**: Native PyTorch Automatic Mixed Precision.
- **Live Real-time Curves**: Embedded Matplotlib graphs displaying Train Loss and Validation Normalized CER directly inside the GUI.

### 4. 📄 End-to-End Full Document & Page OCR Pipeline
- **Khmer Document Line Segmenter**: Automatically slices full-page scans (A4, receipts, certificates, letters) into ordered text lines.
- **Subscript-Safe Morphology**: Specially tuned dilation preserves low-hanging subscripts without chopping them into the next line.
- **Multi-Line Reconstruction**: Transcribes all lines sequentially and reconstructs the full page text with paragraph structure.
- **Export Options**: One-click copy or export to `.txt`.

### 5. 🔍 Khmer-Centric Evaluation & Diagnostics
- **Normalized CER (NCER)**: Evaluates true visual accuracy by ignoring invisible ZWSP and canonical ordering variations.
- **Syllable Cluster Error Rate (SCER)**: Measures error on atomic orthographic clusters (e.g., `[កម្ពុ]`, `[ជា]`, `[ស្រី]`).
- **Glyph Confusion Tracker**: Identifies top misclassified character pairs to inform data augmentation.

### 6. ⚡ Production ONNX Export
- **Dynamic-Width ONNX**: Exports PyTorch models to ONNX with dynamic input widths for deployment in C++, Python, or browser runtimes.

---

## 🚀 Quickstart

### 1. Launch the PyQt5 Desktop Application
```bash
uv run --no-sync python3 run_app.py
```

### 2. Command Line Interface (CLI)

## Audit dataset quality and Khmer Unicode
```bash
uv run --no-sync python3 run_app.py audit --data data/synthetic_train
```

## Recognize full document image (multi-line page)
```bash
uv run --no-sync python3 run_app.py infer --image sample_document.png --ckpt checkpoints/best_model.pth --mode doc
```

## Export model to ONNX
```bash
uv run --no-sync python3 run_app.py export --ckpt checkpoints/best_model.pth --out exports/khmer_ocr.onnx
```

## 1. Generate synthetic dataset from Wikipedia (previously failed, now works seamlessly)                                                    
```bash
uv run --no-sync python run_app.py synth  --wiki --samples 300 --out data/synthetic_wiki
```

## 2. Fine-tune pre-trained model on the harvested Wikipedia dataset with backbone freezing & adaptive LR                                    
```bash
uv run --no-sync python run_app.py train \
--train-dir data/dataset_v2/train \
--val-dir data/dataset_v2/val \
--fine-tune checkpoints/best_model.pth \
--epochs 10 \
--lr 2e-4 \
--lr-scheduler plateau \
--freeze-backbone \
--out-dir checkpoints/wiki_finetuned
```

## 3. Resume training if interrupted (restores exact epoch, optimizer state & best CER)                                                      
```bash
uv run --no-sync python run_app.py train \
      --train-dir data/dataset_v2/train \
      --val-dir data/dataset_v2/val \
      --resume checkpoints/khmer_ocr_v2/latest_model.pth \
      --total-epochs 20 \
      --batch-size 32 \
      --lr-scheduler plateau \
      --early-stopping 5 \
      --out-dir checkpoints/khmer_ocr_v2
```

## To resume seamlessly, run:
```bash
uv run --no-sync python run_app.py train --train-dir data/dataset_v2/train --resume checkpoints/khmer_ocr_v2/interrupted_checkpoint.pth
```

# generate data val, train from txt file

## launched the synthesis of 25,000 balanced Khmer samples into data/dataset_v2 with an automated 10% validation split and 40% clean document contrast.
```bash
uv run --no-sync python run_app.py synth --corpus data/khmer_wiki_corpus.txt -n 25000 --val-split 0.10 --clean-ratio 0.40 -o data/dataset_v2
```
# 4. Download and validate hundreds of authentic Khmer fonts from open-source repositories
```bash
uv run --no-sync python run_app.py download-fonts --out fonts
```

# (Optional) Ingest a custom local ZIP pack of fonts (audits Unicode cmap & deduplicates)
# uv run --no-sync python run_app.py download-fonts --zip path/to/khmer_fonts.zip --out fonts

## 5. Harvest authentic pure Khmer corpus from Wikipedia (km.wikipedia.org)
Harvests authentic Khmer text directly from Khmer Wikipedia with automated foreign language & script sanitization (strips Chinese, Thai, English glosses, and wiki metadata):
```bash
uv run --no-sync python run_app.py wiki \
  --topic history \
  --articles 20 \
  --out data/wiki_khmer_corpus.txt
```
*(Optional: add `--allow-latin` if you explicitly want bilingual sentences containing English words)*

## generate data set from txt with director fonts
```bash
uv run --no-sync python run_app.py synth --corpus data/khmer_wiki_corpus.txt \
  --fonts-dir fonts \
  --samples 35000 \
  --val-split 0.10 \
  --clean-ratio 0.40 \
  --out data/dataset_thousand_fonts
```
*(Tip: Add `--pure-corpus` to strictly sample 100% directly from your corpus text file without mixing RAC dictionary phrases or numeric templates).*
# Reinforcement of rare characters in the training data is complete
```bash
uv run --no-sync python run_app.py boost --data data/dataset_v2/train --threshold 50
```
```bash
uv run --no-sync python run_app.py train --train-dir data/dataset_v2/train --val-dir
data/dataset_v2/val -e 20 --batch-size 32 --lr 5e-4 --lr-scheduler plateau --early-stopping 5 -o
checkpoints/khmer_ocr_v2
```
## Train on completed model with new datasets
```bash
uv run --no-sync python run_app.py train \
      --train-dir data/dataset_thousand_fonts/train \
      --val-dir data/dataset_thousand_fonts/val \
      --fine-tune checkpoints/khmer_ocr_v2/best_model.pth \
      --epochs 15 \
      --batch-size 32 \
      --lr 1.5e-4 \
      --lr-scheduler plateau \
      --early-stopping 5 \
      --out-dir checkpoints/khmer_ocr_multi_font
```

---

## 🧪 Running Automated Tests
```bash
uv run --no-sync pytest tests/ -v
```
agy --conversation=4ee6130b-540b-4c54-adeb-08326dbd9c2c
