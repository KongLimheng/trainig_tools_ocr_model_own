import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_app import build_parser


def test_synth_command_argument_parsing():
    """Verifies that 'synth' parses --samples 300, -n, -c, and --wiki flags."""
    parser = build_parser()

    # User's exact failing command:
    args = parser.parse_args(["synth", "--wiki", "--samples", "300", "--out", "data/synthetic_wiki"])
    assert args.command == "synth"
    assert args.wiki is True
    assert args.count == 300
    assert args.out == "data/synthetic_wiki"

    # Alias with -n, --output, --corpus, --fonts-dir
    args2 = parser.parse_args([
        "synth", "-n", "150", "--output", "data/custom_out",
        "--topic", "all", "--fonts-dir", "fonts", "--corpus", "data/corpus.txt"
    ])
    assert args2.count == 150
    assert args2.out == "data/custom_out"
    assert args2.wiki_topic == "all"
    assert args2.fonts_dir == "fonts"
    assert args2.corpus == "data/corpus.txt"


def test_train_command_argument_parsing():
    """Verifies that 'train' parses --resume, --fine-tune, and continuous training flags."""
    parser = build_parser()

    # Resume command
    args = parser.parse_args([
        "train",
        "--train-dir", "data/train",
        "--resume", "checkpoints/latest_model.pth",
        "-e", "25",
        "-b", "32",
        "--lr", "0.0003",
        "--freeze-backbone",
        "--lr-scheduler", "plateau",
    ])
    assert args.command == "train"
    assert args.train_dir == "data/train"
    assert args.resume == "checkpoints/latest_model.pth"
    assert args.epochs == 25
    assert args.batch_size == 32
    assert args.lr == 0.0003
    assert args.freeze_backbone is True
    assert args.lr_scheduler == "plateau"

    # Fine-tune command with aliases
    args2 = parser.parse_args([
        "train",
        "--data", "data/wiki_corpus",
        "--fine-tune", "checkpoints/best_model.pth",
        "-e", "10",
        "-o", "checkpoints/finetuned",
    ])
    assert args2.train_dir == "data/wiki_corpus"
    assert args2.fine_tune == "checkpoints/best_model.pth"
    assert args2.epochs == 10
    assert args2.out_dir == "checkpoints/finetuned"
    assert args2.lr is None

    # User's exact command with total-epochs
    args3 = parser.parse_args([
        "train",
        "--lr", "2e-4",
        "--train-dir", "data/synthetic_wiki",
        "--resume", "checkpoints/best_model.pth",
        "--epochs", "1000",
        "--total-epochs", "20",
        "--early-stopping", "4",
        "--out-dir", "checkpoints/wiki_finetuned",
    ])
    assert args3.lr == 0.0002
    assert args3.resume == "checkpoints/best_model.pth"
    assert args3.epochs == 1000
    assert args3.total_epochs == 20
    assert args3.early_stopping == 4
    assert args3.out_dir == "checkpoints/wiki_finetuned"


def test_other_subcommand_aliases():
    """Verifies that other subcommands correctly parse aliases."""
    parser = build_parser()

    # wiki parser
    args_wiki = parser.parse_args(["wiki", "--topic", "history", "-n", "20", "-o", "data/out.txt"])
    assert args_wiki.topic == "history"
    assert args_wiki.articles == 20
    assert args_wiki.out == "data/out.txt"
    assert args_wiki.allow_latin is False
    assert args_wiki.min_khmer_ratio == 0.70

    # audit parser
    args_audit = parser.parse_args(["audit", "--dir", "data/my_dataset"])
    assert args_audit.data == "data/my_dataset"

    # infer parser
    args_infer = parser.parse_args(["infer", "-i", "doc.png", "-c", "model.pth", "--export-pdf", "doc.pdf"])
    assert args_infer.image == "doc.png"
    assert args_infer.ckpt == "model.pth"
    assert args_infer.pdf == "doc.pdf"

    # download-fonts parser
    args_fonts = parser.parse_args(["download-fonts", "--fonts-dir", "custom_fonts", "--source", "sbbic"])
    assert args_fonts.out == "custom_fonts"
    assert args_fonts.source == "sbbic"
