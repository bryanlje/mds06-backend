import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class Settings:
    # Paths
    BASE_DIR: Path = Path("/app")
    OUTPUTS_DIR: Path = Path("/tmp/outputs")
    MODELS_DIR: Path = BASE_DIR / "models"
    TRACKNET_REPO: Path = BASE_DIR / "TrackNetV3"
    
    # Model Weights Paths
    TRACKNET_WEIGHTS: str = os.getenv('TRACKNET_WEIGHTS_PATH', str(TRACKNET_REPO / 'ckpts/TrackNet_best.pt'))
    INPAINTNET_WEIGHTS: str = os.getenv('INPAINTNET_WEIGHTS_PATH', str(TRACKNET_REPO / 'ckpts/InpaintNet_best.pt'))
    CONTACT_WEIGHTS: str = os.getenv('CONTACT_WEIGHTS_PATH', str(MODELS_DIR / 'contact_model.pth'))
    SLOWFAST_WEIGHTS: str = os.getenv('SLOWFAST_WEIGHTS_PATH', str(MODELS_DIR / 'slowfast_model.pt'))
    YOLO_WEIGHTS: str = os.getenv('YOLO_WEIGHTS_PATH', str(MODELS_DIR / 'yolo_weights.pt'))
    REID_WEIGHTS: str = os.getenv('REID_WEIGHTS_PATH', str(MODELS_DIR / 'osnet_x1_0_badminton.pt'))

    # Feature Flags
    USE_INPAINTNET: bool = os.getenv('USE_INPAINTNET', 'false').lower() == 'true'
    USE_CONTACT: bool = os.getenv('USE_CONTACT_DETECTION', 'true').lower() == 'true'
    USE_SLOWFAST: bool = os.getenv('USE_SLOWFAST', 'true').lower() == 'true'
    USE_YOLO: bool = os.getenv('USE_YOLO', 'true').lower() == 'true'

    # Inference Configs
    YOLO_CONF: float = 0.50
    STRONGSORT_MAX_AGE: int = 30
    STRONGSORT_N_INIT: int = 12
    STRONGSORT_MAX_IOU_DIST: float = 1.0
    STRONGSORT_MAX_DIST: float = 1.0
    STRONGSORT_NN_BUDGET: int = 240
    STRONGSORT_DET_THRESH: float = 0.4

    # Action Labels
    LABELS: List[str] = field(default_factory=lambda: [
        "block", "clear", "cross_net", "drive", "drop", "jump_smash",
        "lift", "push", "serve", "smash", "straight_net", "tap", "negative"
    ])

settings = Settings()
# Ensure output directory exists
settings.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)