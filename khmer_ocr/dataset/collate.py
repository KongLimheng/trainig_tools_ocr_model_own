"""Dynamic Aspect-Ratio Batch Collation for Variable-Width Text Lines."""

import torch
import torch.nn.functional as F


class DynamicAspectCollate:
    """Pads variable-width images in a batch dynamically to the max width in that batch."""

    def __init__(self, pad_value: float = 1.0):
        """pad_value = 1.0 represents white background when normalized to [-1, 1]."""
        self.pad_value = pad_value

    def __call__(self, batch: list[dict]) -> dict:
        if not batch:
            return {}

        # 1. Find max width among items in this batch
        widths = [item["image"].shape[2] for item in batch]
        max_w = max(widths)
        # Round up to multiple of 8 or 16 for CNN downsampling layers
        if max_w % 16 != 0:
            max_w = ((max_w // 16) + 1) * 16

        padded_images = []
        labels_list = []
        label_lengths = []
        texts = []
        paths = []

        for item in batch:
            img = item["image"]
            c, h, w = img.shape
            pad_w = max_w - w

            if pad_w > 0:
                # Pad on the right side: pad = (pad_left, pad_right, pad_top, pad_bottom)
                img_padded = F.pad(img, (0, pad_w, 0, 0), value=self.pad_value)
            else:
                img_padded = img

            padded_images.append(img_padded)
            lbl = item["label"]
            labels_list.append(lbl)
            label_lengths.append(len(lbl))
            texts.append(item["text"])
            paths.append(item["path"])

        batched_images = torch.stack(padded_images, dim=0)

        # Concatenate labels into a 1D tensor (required for PyTorch torch.nn.CTCLoss)
        concatenated_labels = torch.cat(labels_list, dim=0)
        label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)

        return {
            "images": batched_images,
            "targets": concatenated_labels,
            "target_lengths": label_lengths_tensor,
            "texts": texts,
            "paths": paths,
        }
