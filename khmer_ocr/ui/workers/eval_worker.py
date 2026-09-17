"""Background QThread Worker for Model Evaluation & Diagnostics."""

from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
import torch
from torch.utils.data import DataLoader
from ...models.crnn import KhmerCRNN
from ...vocab.char_map import KhmerCharMap
from ...dataset.folder_dataset import KhmerOCRDataset
from ...dataset.collate import DynamicAspectCollate
from ...metrics.cer import calculate_cer, calculate_exact_match
from ...metrics.normalized_cer import calculate_normalized_cer
from ...metrics.syllable_cer import calculate_syllable_cer
from ...metrics.confusion import KhmerConfusionTracker


class EvaluationWorker(QThread):
    """Background worker for evaluating model performance and computing Khmer metrics."""

    progress_updated = pyqtSignal(int, int)
    eval_finished = pyqtSignal(dict, list, list)
    eval_error = pyqtSignal(str)

    def __init__(
        self,
        checkpoint_path: str | Path,
        dataset_dir: str | Path,
        batch_size: int = 16,
        device: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            device = torch.device(self.device_str)

            # Load checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            char_list = ckpt.get("char_list", [])
            backbone = ckpt.get("backbone", "resnet34")

            char_map = KhmerCharMap(characters=char_list, include_specials=False)
            model = KhmerCRNN(num_classes=len(char_map), backbone_type=backbone)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()

            # Load dataset
            dataset = KhmerOCRDataset(data_dir=self.dataset_dir, char_map=char_map)
            if len(dataset) == 0:
                raise ValueError("No valid labeled images found in dataset directory!")

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=DynamicAspectCollate(),
            )

            all_targets = []
            all_preds = []
            sample_details = []
            confusion_tracker = KhmerConfusionTracker()

            total_batches = len(loader)
            with torch.no_grad():
                for b_idx, batch in enumerate(loader):
                    if self._stop_requested:
                        break

                    images = batch["images"].to(device)
                    raw_texts = batch["texts"]
                    paths = batch["paths"]

                    log_probs = model(images)
                    decoded = model.decode_greedy(log_probs)

                    for idx, seq in enumerate(decoded):
                        pred_str = char_map.decode(seq)
                        targ_str = raw_texts[idx]

                        all_targets.append(targ_str)
                        all_preds.append(pred_str)
                        confusion_tracker.update(targ_str, pred_str)

                        sample_details.append({
                            "path": paths[idx],
                            "target": targ_str,
                            "pred": pred_str,
                            "match": targ_str == pred_str,
                        })

                    self.progress_updated.emit(b_idx + 1, total_batches)

            # Metrics
            cer = calculate_cer(all_targets, all_preds)
            norm_cer = calculate_normalized_cer(all_targets, all_preds)
            scer = calculate_syllable_cer(all_targets, all_preds)
            exact = calculate_exact_match(all_targets, all_preds)

            metrics = {
                "cer": cer,
                "normalized_cer": norm_cer,
                "syllable_cer": scer,
                "exact_match": exact,
                "total_samples": len(all_targets),
            }

            top_confusions = confusion_tracker.top_confusions(15)

            self.eval_finished.emit(metrics, sample_details, top_confusions)

        except Exception as e:
            self.eval_error.emit(str(e))
