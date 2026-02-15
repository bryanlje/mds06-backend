###############################################################################
# Singleton Model Manager.
###############################################################################

import threading
import torch
from ultralytics import YOLO
from boxmot import StrongSort
from pathlib import Path
from app.config import settings
from app.services.contact import ContactDetectionCNN
from app.services.action import load_slowfast_classifier

class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = None
        self.strongsort_tracker = None
        self.contact_model = None
        self.slowfast_model = None
        
        self.yolo_lock = threading.Lock()
        self.contact_lock = threading.Lock()
        self.slowfast_lock = threading.Lock()
        
        self._load_models()
        self._initialized = True

    def _load_models(self):
        print(f"📦 Loading models on {self.device}...")

        if settings.USE_YOLO:
            print("  → YOLO & StrongSort")
            self.yolo_model = YOLO(settings.YOLO_WEIGHTS)
            self.strongsort_tracker = StrongSort(
                reid_weights=Path(settings.REID_WEIGHTS),
                device=0 if torch.cuda.is_available() else 'cpu',
                half=True if torch.cuda.is_available() else False,
                max_age=settings.STRONGSORT_MAX_AGE,
                n_init=settings.STRONGSORT_N_INIT,
                max_iou_dist=settings.STRONGSORT_MAX_IOU_DIST,
                max_cos_dist=settings.STRONGSORT_MAX_DIST,
                nn_budget=settings.STRONGSORT_NN_BUDGET,
                det_thresh=settings.STRONGSORT_DET_THRESH
            )

        if settings.USE_CONTACT:
            print("  → Contact Detection")
            self.contact_model = ContactDetectionCNN(9, 64)
            self.contact_model.load_state_dict(torch.load(settings.CONTACT_WEIGHTS, map_location=self.device))
            self.contact_model.to(self.device).eval()

        if settings.USE_SLOWFAST:
            print("  → SlowFast")
            self.slowfast_model = load_slowfast_classifier(settings.LABELS, settings.SLOWFAST_WEIGHTS, self.device)

        print("✅ Models loaded.")

model_manager = ModelManager()