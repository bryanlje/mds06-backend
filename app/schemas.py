from dataclasses import dataclass
from typing import Optional, Any, Sequence, Tuple, List

@dataclass
class ProcessingResult:
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass
class YoloCfg:
    conf: float = 0.50
    iou: float = 0.8
    imgsz: int = 1280
    max_det: int = 2
    classes: Optional[Sequence[int]] = None
    agnostic_nms: bool = False
    verbose: bool = False

@dataclass
class ContactConfig:
    window_size: int = 21
    temporal_stride: int = 5
    positive_window: int = 4
    compute_velocity: bool = True
    compute_acceleration: bool = True
    compute_direction: bool = True
    smooth_window: int = 5

@dataclass
class SlowFastInferCfg:
    slow_t: int = 8
    alpha: int = 4
    side: int = 224
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45)
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
    bbox_margin: float = 1.3
    bbox_ema: float = 0.8

@dataclass
class InferenceCfg:
    labels: List[str]
    sf: SlowFastInferCfg
    n_before_after: int = 18
    search_span: int = 3