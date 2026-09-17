"""Unit tests for Rare Character & Cluster Coverage Booster."""

from pathlib import Path
from khmer_ocr.dataset.coverage_booster import KhmerCoverageBooster, RARE_TARGET_CHARACTERS


def test_coverage_audit_and_phrase_generation():
    booster = KhmerCoverageBooster()
    # Test phrase generation for rare independent vowel (e.g. ឪ)
    phrases_iv = booster.generate_targeted_phrases("ឪ", count=3)
    assert len(phrases_iv) == 3
    for p in phrases_iv:
        assert "ឪ" in p

    # Test phrase generation for rare diacritic (e.g. ៏)
    phrases_dia = booster.generate_targeted_phrases("៏", count=3)
    assert len(phrases_dia) == 3
    for p in phrases_dia:
        assert "៏" in p


def test_coverage_booster_dataset(tmp_path: Path):
    booster = KhmerCoverageBooster()
    dataset_dir = tmp_path / "mini_ds"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)

    # Create dummy dataset labels with common characters only
    labels_file = dataset_dir / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write("001.jpg\tកម្ពុជាជាតិសាសនាព្រះមហាក្សត្រ\n")
        f.write("002.jpg\tរាជធានីភ្នំពេញសាលាក្រុង\n")

    # Run audit
    counts = booster.audit_coverage(labels_file)
    assert counts.get("ក", 0) > 0
    # Rare characters like ឦ, ឩ, ឪ should be 0
    assert counts.get("ឦ", 0) == 0

    # Boost dataset with a small threshold and max samples
    res = booster.boost_dataset(dataset_dir, min_threshold=2)
    assert res["boosted_characters"] > 0
    assert res["samples_added"] > 0

    # Verify labels.txt was appended with new lines containing rare characters
    with open(labels_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    assert len(all_lines) > 2

    # Verify generated images exist
    generated_images = list(images_dir.glob("boost_*.jpg"))
    assert len(generated_images) == res["samples_added"]
