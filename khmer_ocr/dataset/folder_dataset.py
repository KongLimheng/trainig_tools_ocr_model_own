"""PyTorch Dataset for Khmer OCR Line Recognition."""

from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from ..vocab.char_map import KhmerCharMap
from ..normalizer import normalize_khmer_text


class KhmerOCRDataset(Dataset):
    """Loads line images and Khmer transcriptions from a directory containing labels.txt."""

    def __init__(
        self,
        data_dir: str | Path,
        labels_filename: str = "labels.txt",
        char_map: KhmerCharMap | None = None,
        img_height: int = 48,
        channels: int = 1,
        normalize_text: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.char_map = char_map or KhmerCharMap()
        self.img_height = img_height
        self.channels = channels
        self.normalize_text = normalize_text
        self.transform = transform

        labels_path = self.data_dir / labels_filename
        if not labels_path.exists() and self.data_dir.is_file():
            labels_path = self.data_dir
            self.data_dir = self.data_dir.parent

        self.images_dir = self.data_dir / "images"
        if not self.images_dir.exists():
            self.images_dir = self.data_dir

        self.samples: list[tuple[Path, str]] = []
        self._load_labels(labels_path)

    def _load_labels(self, labels_path: Path) -> None:
        """Reads labels file."""
        if not labels_path.exists():
            return

        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                img_name, text = parts[0], parts[1]
                img_path = self.images_dir / img_name
                if not img_path.exists():
                    img_path = self.data_dir / img_name

                if img_path.exists():
                    if self.normalize_text:
                        text = normalize_khmer_text(text, strip_zwsp=True)
                    if text:
                        self.samples.append((img_path, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, text = self.samples[idx]

        mode = "L" if self.channels == 1 else "RGB"
        with Image.open(img_path) as img:
            img = img.convert(mode)
            w, h = img.size

            # Resize keeping aspect ratio to img_height
            aspect_ratio = w / max(1, h)
            new_w = max(16, int(self.img_height * aspect_ratio))
            img = img.resize((new_w, self.img_height), Image.Resampling.BILINEAR)

            # Convert to tensor [C, H, W] normalized to [0, 1]
            img_tensor = T.functional.to_tensor(img)
            # Normalize to [-1, 1]
            img_tensor = (img_tensor - 0.5) / 0.5

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        encoded_label = self.char_map.encode(text)

        return {
            "image": img_tensor,
            "label": torch.tensor(encoded_label, dtype=torch.long),
            "text": text,
            "path": str(img_path),
        }
