#!/usr/bin/env python3
"""Unified Entry Point for KhmerOCR Tools (PyQt5 GUI & CLI)."""

import sys
import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser with aliases."""
    parser = argparse.ArgumentParser(
        description="Khmer AI OCR Training & Tooling Suite")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: gui (default)
    subparsers.add_parser("gui", help="Launch PyQt5 Desktop Application")

    # Command: synth
    synth_parser = subparsers.add_parser(
        "synth", help="Generate synthetic Khmer dataset")
    synth_parser.add_argument(
        "-o", "--out", "--output",
        dest="out",
        default="data/synthetic_train",
        help="Output directory (aliases: -o, --output)"
    )
    synth_parser.add_argument(
        "-c", "--count", "--samples", "-n",
        dest="count",
        type=int,
        default=500,
        help="Number of samples to generate (aliases: --samples, -n, -c)"
    )
    synth_parser.add_argument("--height", type=int,
                              default=48, help="Line image height in pixels")
    synth_parser.add_argument(
        "--no-augment", action="store_true", help="Disable degradations")
    synth_parser.add_argument("--wiki", action="store_true",
                              help="Harvest corpus from Khmer Wikipedia before generating")
    synth_parser.add_argument(
        "--wiki-topic", "--topic",
        dest="wiki_topic",
        default=None,
        help="Specific topic or keyword to harvest from Wikipedia ('all' for curated topics)"
    )
    synth_parser.add_argument(
        "--wiki-articles",
        type=int,
        default=15,
        help="Number of Wikipedia articles to fetch when --wiki is enabled"
    )
    synth_parser.add_argument(
        "--val-split", "--val-ratio",
        dest="val_ratio",
        type=float,
        default=0.0,
        help="Ratio of samples to reserve for validation split (e.g. 0.1 for 10% val split)"
    )
    synth_parser.add_argument(
        "--clean-ratio",
        dest="clean_ratio",
        type=float,
        default=0.40,
        help="Ratio of samples rendered in high-contrast clean document style (default: 0.40)"
    )
    synth_parser.add_argument(
        "--corpus",
        dest="corpus",
        default=None,
        help="Path to text corpus file to sample from (e.g. data/khmer_wiki_corpus.txt)"
    )

    # Command: audit
    audit_parser = subparsers.add_parser(
        "audit", help="Audit dataset quality and Khmer Unicode")
    audit_parser.add_argument("-d", "--data", "--dir", dest="data",
                              required=True, help="Dataset directory containing labels.txt")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train Khmer OCR model")
    train_parser.add_argument("-d", "--train-dir", "--data", dest="train_dir",
                              required=True, help="Training dataset directory")
    train_parser.add_argument(
        "--val-dir", default=None, help="Validation dataset directory")
    train_parser.add_argument("-e", "--epochs", dest="epochs",
                              type=int, default=15, help="Number of training epochs")
    train_parser.add_argument(
        "-b", "--batch-size", dest="batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 5e-4, or preserve checkpoint LR when resuming if omitted)")
    train_parser.add_argument("--backbone", default="resnet34",
                              choices=["resnet34", "mobilenet"], help="Backbone")
    train_parser.add_argument("-o", "--out-dir", dest="out_dir",
                              default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument(
        "--resume", default=None, help="Resume interrupted training from checkpoint (.pth)")
    train_parser.add_argument("--fine-tune", "--ckpt", dest="fine_tune",
                              default=None, help="Fine-tune pre-trained checkpoint on new dataset")
    train_parser.add_argument("--freeze-backbone", action="store_true",
                              help="Freeze CNN backbone to train RNN & CTC head only")
    train_parser.add_argument("--lr-scheduler", default="plateau", choices=[
                              "plateau", "cosine", "none"], help="Adaptive learning rate scheduler")
    train_parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        type=int,
        default=5,
        help="Epochs without validation CER improvement before stopping (0 to disable, default: 5)"
    )
    train_parser.add_argument(
        "--total-epochs",
        dest="total_epochs",
        type=int,
        default=None,
        help="Target total epochs to reach (automatically calculates remaining epochs when resuming)"
    )

    # Command: eval
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate model with Khmer metrics")
    eval_parser.add_argument("-c", "--ckpt", dest="ckpt",
                             required=True, help="Model checkpoint path (.pth)")
    eval_parser.add_argument(
        "-t", "--test-dir", dest="test_dir", required=True, help="Test dataset directory")

    # Command: export
    export_parser = subparsers.add_parser(
        "export", help="Export PyTorch model to dynamic ONNX")
    export_parser.add_argument(
        "-c", "--ckpt", dest="ckpt", required=True, help="Model checkpoint path (.pth)")
    export_parser.add_argument("-o", "--out", "--output", dest="out",
                               default="exports/khmer_ocr.onnx", help="Output ONNX path")

    # Command: infer
    infer_parser = subparsers.add_parser(
        "infer", help="Recognize Khmer text from image or document")
    infer_parser.add_argument(
        "-i", "--image", dest="image", required=True, help="Input image path")
    infer_parser.add_argument("-c", "--ckpt", dest="ckpt",
                              default="checkpoints/best_model.pth", help="Model checkpoint (.pth)")
    infer_parser.add_argument(
        "--mode", default="doc", choices=["doc", "line"], help="doc (multi-line page) or line")
    infer_parser.add_argument(
        "--deskew", action="store_true", help="Auto-deskew and remove shadows before OCR")
    infer_parser.add_argument(
        "--wordseg", action="store_true", help="Segment continuous text with spaces")
    infer_parser.add_argument(
        "--spell", action="store_true", help="Auto-correct common OCR spelling mistakes")
    infer_parser.add_argument("--pdf", "--export-pdf", dest="pdf", default=None,
                              help="Export output as Searchable PDF (e.g. output.pdf)")

    # Command: boost
    boost_parser = subparsers.add_parser(
        "boost", help="Audit and boost rare glyph coverage in dataset")
    boost_parser.add_argument("-d", "--data", "--dir", dest="data",
                              required=True, help="Dataset directory containing labels.txt")
    boost_parser.add_argument("-t", "--threshold", dest="threshold",
                              type=int, default=30, help="Minimum occurrences for rare characters")

    # Command: wiki
    wiki_parser = subparsers.add_parser(
        "wiki", help="Harvest massive authentic text from Khmer Wikipedia (km.wikipedia.org)")
    wiki_parser.add_argument("-t", "--topic", dest="topic", default=None,
                             help="Khmer topic keyword or domain (e.g. 'កម្ពុជា', 'អង្គរវត្ត')")
    wiki_parser.add_argument("-n", "--articles", "--count", dest="articles",
                             type=int, default=15, help="Number of articles to harvest")
    wiki_parser.add_argument(
        "--random", action="store_true", help="Crawl random articles across Wikipedia")
    wiki_parser.add_argument("-o", "--out", "--output", dest="out",
                             default="data/wikipedia_khmer_corpus.txt", help="Output .txt file path")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no subcommand provided, default to launching PyQt5 GUI
    if args.command is None or args.command == "gui":
        from khmer_ocr.ui.main_window import run_studio_app
        run_studio_app()
        return

    if args.command == "synth":
        from khmer_ocr.synth.generator import KhmerDatasetGenerator

        sampler = None
        if args.wiki:
            from khmer_ocr.dataset.wikipedia_loader import KhmerWikipediaLoader
            from khmer_ocr.synth.corpus_sampler import KhmerCorpusSampler
            loader = KhmerWikipediaLoader()
            print("Harvesting authentic Khmer text from km.wikipedia.org...")
            articles_count = getattr(args, "wiki_articles", 15)
            if args.wiki_topic and args.wiki_topic.lower() != "all":
                sentences = loader.harvest_from_search(
                    args.wiki_topic, max_articles=articles_count)
            else:
                sentences = loader.harvest_from_topics(
                    max_articles=articles_count)
            print(f"Harvested {len(sentences):,} sentences from Wikipedia.")
            if sentences:
                sampler = KhmerCorpusSampler(custom_texts=sentences)
            else:
                print(
                    "Notice: No Wikipedia sentences were harvested. Falling back to default corpus sampler.")
        elif getattr(args, "corpus", None):
            from khmer_ocr.synth.corpus_sampler import KhmerCorpusSampler
            corpus_path = Path(args.corpus)
            if corpus_path.exists():
                sentences = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                print(f"Loaded {len(sentences):,} sentences from corpus: {args.corpus}")
                sampler = KhmerCorpusSampler(custom_texts=sentences)
            else:
                print(f"Warning: Corpus file '{args.corpus}' not found. Falling back to default sampler.")

        print(f"Generating {args.count} samples into '{args.out}'...")
        gen = KhmerDatasetGenerator(
            target_height=args.height, corpus_sampler=sampler)
        labels_file = gen.generate_batch(
            output_dir=args.out,
            num_samples=args.count,
            augment=not args.no_augment,
            val_ratio=args.val_ratio,
            clean_ratio=args.clean_ratio,
            progress_callback=lambda cur, tot, msg: print(
                f"[{cur}/{tot}] {msg}", end="\r"),
        )
        print(f"\nDone! Labels saved to {labels_file}")

    elif args.command == "audit":
        from khmer_ocr.dataset.data_audit import audit_dataset
        print(f"Auditing dataset at: {args.data}")
        results = audit_dataset(args.data)
        print("=" * 50)
        print(f"Total Samples:            {results['total_samples']}")
        print(f"Missing Images:           {results['missing_images']}")
        print(f"Invisible Chars (ZWSP):   {results['invisible_char_samples']}")
        print(f"Non-canonical sequences:  {results['non_canonical_samples']}")
        print(f"Unique Characters:        {results['unique_characters']}")
        print(f"OOV Characters:           {results['oov_character_count']}")
        print("=" * 50)

    elif args.command == "train":
        import torch
        from torch.utils.data import DataLoader
        from khmer_ocr.vocab.char_map import KhmerCharMap
        from khmer_ocr.models.crnn import KhmerCRNN
        from khmer_ocr.dataset.folder_dataset import KhmerOCRDataset
        from khmer_ocr.dataset.collate import DynamicAspectCollate
        from khmer_ocr.training.trainer import KhmerOCRTrainer

        char_map = KhmerCharMap()
        train_ds = KhmerOCRDataset(args.train_dir, char_map=char_map)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=DynamicAspectCollate())

        val_loader = None
        if args.val_dir:
            val_ds = KhmerOCRDataset(args.val_dir, char_map=char_map)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=DynamicAspectCollate()
            )

        ckpt_to_load = args.resume or args.fine_tune
        is_resume = bool(args.resume)
        start_epoch = 0
        best_val_cer = float("inf")
        resume_optimizer_state = None

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if ckpt_to_load:
            from khmer_ocr.models.checkpoint_manager import load_and_adapt_checkpoint, inspect_checkpoint
            mode_str = "resuming" if is_resume else "fine-tuning"
            print(f"Loading checkpoint for {mode_str}: {ckpt_to_load}")
            model, adapt_info = load_and_adapt_checkpoint(
                checkpoint_path=ckpt_to_load,
                target_char_map=char_map,
                target_backbone=args.backbone,
                device=device_str,
            )
            print(
                f"Intelligent vocabulary adaptation: {adapt_info['transferred_chars']} transferred, {adapt_info['new_chars']} new characters.")
            raw_ckpt = adapt_info.get("raw_checkpoint", {})
            if is_resume:
                start_epoch = raw_ckpt.get("epoch", 0)
                best_val_cer = raw_ckpt.get("val_metrics", {}).get(
                    "norm_cer", float("inf"))
                resume_optimizer_state = raw_ckpt.get(
                    "optimizer_state_dict", None)
                print(
                    f"Resuming run from epoch {start_epoch} (Previous Best Norm CER: {best_val_cer:.2%})...")
            else:
                print(
                    f"Fine-tuning model on new dataset with pre-trained weights from epoch {raw_ckpt.get('epoch', 0)}...")
        else:
            model = KhmerCRNN(num_classes=len(char_map),
                              backbone_type=args.backbone)

        if args.lr is not None:
            lr_to_use = args.lr
            override_lr = True
        else:
            if is_resume:
                lr_to_use = None
                override_lr = False
            else:
                lr_to_use = 5e-4
                override_lr = True

        trainer = KhmerOCRTrainer(
            model=model,
            char_map=char_map,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=lr_to_use,
            output_dir=args.out_dir,
            start_epoch=start_epoch,
            best_val_cer=best_val_cer,
            resume_optimizer_state=resume_optimizer_state,
            lr_scheduler_type=args.lr_scheduler if args.lr_scheduler != "none" else None,
            freeze_backbone=args.freeze_backbone,
            override_lr=override_lr,
            early_stopping_patience=args.early_stopping if args.early_stopping > 0 else None,
        )

        def on_epoch(ep, total_ep, loss, metrics):
            current_lr = metrics.get('lr', trainer.get_current_lr())
            lr_str = f" - LR: {current_lr:.6f}" if current_lr is not None else ""
            if metrics.get("has_val", False):
                print(
                    f"Epoch {ep}/{total_ep} - Train Loss: {loss:.4f} - Val Norm CER: {metrics.get('norm_cer', 0.0):.2%} - Val Exact: {metrics.get('exact_match', 0.0):.1%}{lr_str}")
            else:
                print(f"Epoch {ep}/{total_ep} - Train Loss: {loss:.4f} (No Val Set){lr_str}")

        epochs_to_run = args.epochs
        if args.total_epochs is not None:
            if start_epoch >= args.total_epochs:
                print(f"Notice: Checkpoint is already at epoch {start_epoch}, which meets or exceeds --total-epochs {args.total_epochs}.")
                epochs_to_run = 0
            else:
                epochs_to_run = args.total_epochs - start_epoch
                print(f"Target total epochs: {args.total_epochs} (resuming from epoch {start_epoch}, running {epochs_to_run} remaining epochs).")

        if epochs_to_run <= 0:
            print("No additional epochs to train. Checkpoint is up to date!")
            return

        print("Starting training...")
        best_path = trainer.train(epochs=epochs_to_run, epoch_callback=on_epoch)
        print(f"Training complete! Best model: {best_path}")

    elif args.command == "eval":
        from khmer_ocr.ui.workers.eval_worker import EvaluationWorker
        print(f"Evaluating checkpoint {args.ckpt} on {args.test_dir}...")
        worker = EvaluationWorker(
            checkpoint_path=args.ckpt, dataset_dir=args.test_dir)

        def on_eval_finished(metrics, sample_details, top_confusions):
            print("=" * 50)
            print(f"Total Tested Samples:        {metrics['total_samples']}")
            print(
                f"Normalized CER (True Khmer): {metrics['normalized_cer'] * 100:.2f}%")
            print(f"Standard CER:                {metrics['cer'] * 100:.2f}%")
            print(
                f"Syllable Cluster Error Rate: {metrics['syllable_cer'] * 100:.2f}%")
            print(
                f"Exact Sequence Match:        {metrics['exact_match'] * 100:.2f}%")
            print("=" * 50)
            if top_confusions:
                print("Top Confused Glyphs:")
                for c in top_confusions[:5]:
                    print(
                        f"  '{c['ground_truth']}' -> predicted as '{c['predicted']}' ({c['count']} times)")

        worker.eval_finished.connect(on_eval_finished)
        worker.run()

    elif args.command == "export":
        from khmer_ocr.export.onnx_exporter import export_to_onnx
        print(f"Exporting {args.ckpt} to dynamic ONNX: {args.out}...")
        out_path = export_to_onnx(args.ckpt, args.out)
        print(f"Export succeeded! Saved to {out_path}")

    elif args.command == "boost":
        from khmer_ocr.dataset.coverage_booster import KhmerCoverageBooster
        print(
            f"Auditing and boosting dataset at: {args.data} (threshold={args.threshold})...")
        booster = KhmerCoverageBooster()
        res = booster.boost_dataset(
            args.data,
            min_threshold=args.threshold,
            progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"),
        )
        print(
            f"Done! Reinforced {res['boosted_characters']} rare glyphs with {res['samples_added']} synthesized samples.")

    elif args.command == "wiki":
        from khmer_ocr.dataset.wikipedia_loader import KhmerWikipediaLoader
        loader = KhmerWikipediaLoader()
        print("Connecting to km.wikipedia.org...")
        if args.random:
            print(f"Crawling {args.articles} random articles...")
            sentences = loader.harvest_from_random(
                count=args.articles,
                progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"),
            )
        elif args.topic:
            print(
                f"Searching and harvesting articles for topic: '{args.topic}'...")
            sentences = loader.harvest_from_search(
                query=args.topic,
                max_articles=args.articles,
                progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"),
            )
        else:
            print(
                f"Harvesting curated Cambodian topics ({args.articles} articles)...")
            sentences = loader.harvest_from_topics(
                max_articles=args.articles,
                progress_callback=lambda c, t, m: print(f"[{c}/{t}] {m}"),
            )

        count = loader.save_corpus(sentences, args.out)
        print(
            f"Done! Successfully harvested {count:,} authentic Khmer sentences into: {args.out}")

    elif args.command == "infer":
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from khmer_ocr.models.crnn import KhmerCRNN
        from khmer_ocr.vocab.char_map import KhmerCharMap
        from khmer_ocr.pipeline.segmenter import KhmerDocumentLineSegmenter

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        char_map = KhmerCharMap(
            characters=ckpt["char_list"], include_specials=False)
        model = KhmerCRNN(num_classes=len(char_map),
                          backbone_type=ckpt["backbone"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        img = Image.open(args.image)

        # Preprocessing: auto-deskew & shadow removal
        if args.deskew:
            from khmer_ocr.pipeline.preprocessor import KhmerDocumentPreprocessor
            preprocessor = KhmerDocumentPreprocessor()
            img, meta = preprocessor.preprocess_document(
                img, auto_deskew=True, remove_shadow=True)
            if abs(meta.get("skew_angle_deg", 0.0)) > 0.2:
                print(
                    f"[Preprocessing] Corrected skew angle: {meta['skew_angle_deg']:.1f}°")

        # Post-processing modules
        word_segmenter = None
        spell_corrector = None
        if args.wordseg or args.spell:
            from khmer_ocr.postprocess.word_segmenter import KhmerWordSegmenter
            from khmer_ocr.postprocess.spell_corrector import KhmerSpellCorrector
            word_segmenter = KhmerWordSegmenter()
            if args.spell:
                spell_corrector = KhmerSpellCorrector(segmenter=word_segmenter)

        transcriptions = []
        bboxes = []

        if args.mode == "doc":
            segmenter = KhmerDocumentLineSegmenter()
            bboxes, line_crops, _ = segmenter.segment(img)
            print(f"Detected {len(bboxes)} text lines in document:")
            print("=" * 50)
            for i, crop in enumerate(line_crops):
                im = crop.convert("L")
                w, h = im.size
                new_w = max(16, int(48 * (w / max(1, h))))
                im = im.resize((new_w, 48), Image.Resampling.BILINEAR)
                tensor = ((T.functional.to_tensor(im) - 0.5) /
                          0.5).unsqueeze(0).to(device)
                with torch.no_grad():
                    log_probs = model(tensor)
                    decoded = model.decode_greedy(log_probs)
                    text = char_map.decode(decoded[0]) if decoded else ""

                if spell_corrector:
                    text, _ = spell_corrector.correct_sentence(text)
                if word_segmenter and args.wordseg:
                    text = word_segmenter.segment_to_string(
                        text, delimiter=" ")

                transcriptions.append(text)
                print(f"[Line {i+1}]: {text}")
            print("=" * 50)
        else:
            im = img.convert("L")
            w, h = im.size
            new_w = max(16, int(48 * (w / max(1, h))))
            im = im.resize((new_w, 48), Image.Resampling.BILINEAR)
            tensor = ((T.functional.to_tensor(im) - 0.5) /
                      0.5).unsqueeze(0).to(device)
            with torch.no_grad():
                log_probs = model(tensor)
                decoded = model.decode_greedy(log_probs)
                text = char_map.decode(decoded[0]) if decoded else ""

            if spell_corrector:
                text, _ = spell_corrector.correct_sentence(text)
            if word_segmenter and args.wordseg:
                text = word_segmenter.segment_to_string(text, delimiter=" ")

            bboxes = [(0, 0, img.width, img.height)]
            transcriptions = [text]
            print("Recognized Text:")
            print(text)

        # Searchable PDF export
        if args.pdf:
            from khmer_ocr.export.searchable_pdf import SearchablePDFExporter
            pdf_exporter = SearchablePDFExporter()
            pdf_path = pdf_exporter.export(
                img, bboxes, transcriptions, args.pdf)
            print(f"[Searchable PDF] Successfully exported to: {pdf_path}")


if __name__ == "__main__":
    main()
