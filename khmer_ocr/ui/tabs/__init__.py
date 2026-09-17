"""UI Tabs Package for Khmer OCR Studio."""

from .tab_synth import TabSynthStudio
from .tab_audit import TabDatasetAudit
from .tab_train import TabTrainingStudio
from .tab_eval import TabEvalStudio
from .tab_infer import TabInferAndExport

__all__ = [
    "TabSynthStudio",
    "TabDatasetAudit",
    "TabTrainingStudio",
    "TabEvalStudio",
    "TabInferAndExport",
]
