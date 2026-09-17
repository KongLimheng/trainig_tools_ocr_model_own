"""Unit tests for checkpoint inspection, vocabulary adaptation, and continuous training."""

import tempfile
from pathlib import Path
import pytest
import torch
from khmer_ocr.models.crnn import KhmerCRNN
from khmer_ocr.models.checkpoint_manager import inspect_checkpoint, load_and_adapt_checkpoint
from khmer_ocr.vocab.char_map import KhmerCharMap
from khmer_ocr.training.trainer import KhmerOCRTrainer


def test_checkpoint_inspection_and_exact_resume():
    """Tests saving a checkpoint, inspecting it, and exact resume loading."""
    char_map = KhmerCharMap(characters=["ក", "ខ", "គ", "ឃ", "ង", "ា", "ិ"])
    model = KhmerCRNN(num_classes=len(char_map), backbone_type="resnet34")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pth"
        torch.save({
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"mock_opt": 123},
            "char_list": char_map.char_list,
            "backbone": "resnet34",
            "val_metrics": {"norm_cer": 0.045, "exact_match": 0.88},
        }, ckpt_path)

        # 1. Inspect checkpoint
        info = inspect_checkpoint(ckpt_path)
        assert info["epoch"] == 5
        assert info["char_count"] == len(char_map.char_list)
        assert info["backbone"] == "resnet34"
        assert info["norm_cer"] == 0.045
        assert info["has_optimizer"] is True

        # 2. Exact resume
        adapted_model, adapt_info = load_and_adapt_checkpoint(
            checkpoint_path=ckpt_path,
            target_char_map=char_map,
            device="cpu",
        )
        assert adapt_info["mode"] == "exact_match"
        assert adapt_info["transferred_chars"] == len(char_map)
        assert adapt_info["new_chars"] == 0


def test_vocabulary_expansion_head_adaptation():
    """Tests adapting a model from N classes to N + M classes without losing learned weights."""
    # Pre-trained on 5 basic consonants
    old_chars = ["ក", "ខ", "គ", "ឃ", "ង"]
    old_char_map = KhmerCharMap(characters=old_chars)
    old_model = KhmerCRNN(num_classes=len(old_char_map), backbone_type="resnet34")

    # Set specific identifiable weights for old linear head
    with torch.no_grad():
        old_model.fc.weight.fill_(2.5)
        old_model.fc.bias.fill_(1.2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "pretrained.pth"
        torch.save({
            "epoch": 10,
            "model_state_dict": old_model.state_dict(),
            "char_list": old_char_map.char_list,
            "backbone": "resnet34",
            "val_metrics": {"norm_cer": 0.02},
        }, ckpt_path)

        # New dataset with rare characters added: ឫ, ឬ, ឦ
        new_chars = ["ក", "ខ", "គ", "ឃ", "ង", "ឫ", "ឬ", "ឦ"]
        new_char_map = KhmerCharMap(characters=new_chars)
        assert len(new_char_map) > len(old_char_map)

        adapted_model, adapt_info = load_and_adapt_checkpoint(
            checkpoint_path=ckpt_path,
            target_char_map=new_char_map,
            device="cpu",
        )

        assert adapt_info["mode"] == "adapted"
        assert adapt_info["transferred_chars"] == len(old_char_map)
        assert adapt_info["new_chars"] == 3
        assert "ឫ" in adapt_info["new_char_list"]
        assert "ឬ" in adapt_info["new_char_list"]
        assert "ឦ" in adapt_info["new_char_list"]

        # Check that existing characters retained their pre-trained weights
        ka_idx = new_char_map.char_to_idx["ក"]
        assert torch.allclose(adapted_model.fc.weight[ka_idx], torch.tensor(2.5), atol=1e-5)
        assert torch.allclose(adapted_model.fc.bias[ka_idx], torch.tensor(1.2), atol=1e-5)

        # Check forward pass with adapted model
        dummy = torch.randn(2, 1, 48, 128)
        with torch.no_grad():
            out = adapted_model(dummy)
        assert out.shape[2] == len(new_char_map)


def test_backbone_freezing():
    """Tests freezing and unfreezing of CNN backbone parameters."""
    char_map = KhmerCharMap()
    model = KhmerCRNN(num_classes=len(char_map), backbone_type="resnet34")

    # Initially unfrozen
    assert not model.is_backbone_frozen()
    assert all(p.requires_grad for p in model.backbone.parameters())

    # Freeze backbone
    model.freeze_backbone(True)
    assert model.is_backbone_frozen()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # RNN and FC head should remain trainable
    assert all(p.requires_grad for p in model.rnn.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())

    # Unfreeze backbone
    model.freeze_backbone(False)
    assert not model.is_backbone_frozen()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_adaptive_lr_scheduler_stepping():
    """Tests that ReduceLROnPlateau drops learning rate on plateaus in trainer."""
    char_map = KhmerCharMap(characters=["ក", "ខ"])
    model = KhmerCRNN(num_classes=len(char_map), backbone_type="resnet34")

    trainer = KhmerOCRTrainer(
        model=model,
        char_map=char_map,
        train_loader=[],
        learning_rate=1e-3,
        lr_scheduler_type="plateau",
        device="cpu",
    )

    assert trainer.scheduler is not None
    assert trainer.get_current_lr() == 1e-3

    # Simulate 4 plateaus with static metric (patience=2, factor=0.5)
    for _ in range(4):
        trainer.scheduler.step(0.5)

    # LR should have dropped from 1e-3 to 5e-4
    assert trainer.get_current_lr() < 1e-3


def test_resume_optimizer_learning_rate_override():
    """Tests that user-specified learning rate overrides saved optimizer LR when resuming."""
    char_map = KhmerCharMap(characters=["ក", "ខ"])
    model = KhmerCRNN(num_classes=len(char_map), backbone_type="resnet34")

    # Create real optimizer state dict with decayed lr (6.25e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=6.25e-6)
    saved_opt_state = opt.state_dict()

    trainer = KhmerOCRTrainer(
        model=model,
        char_map=char_map,
        train_loader=[],
        learning_rate=2e-4,
        resume_optimizer_state=saved_opt_state,
        override_lr=True,
        device="cpu",
    )

    assert trainer.get_current_lr() == 2e-4


def test_resume_optimizer_learning_rate_preserve():
    """Tests that checkpoint learning rate is preserved if override_lr is False."""
    char_map = KhmerCharMap(characters=["ក", "ខ"])
    model = KhmerCRNN(num_classes=len(char_map), backbone_type="resnet34")

    # Create real optimizer state dict with decayed lr (6.25e-6)
    opt = torch.optim.AdamW(model.parameters(), lr=6.25e-6)
    saved_opt_state = opt.state_dict()

    trainer = KhmerOCRTrainer(
        model=model,
        char_map=char_map,
        train_loader=[],
        learning_rate=None,
        resume_optimizer_state=saved_opt_state,
        override_lr=False,
        device="cpu",
    )

    assert trainer.get_current_lr() == pytest.approx(6.25e-6)
