################################################################################
# Helper functions for drawing, math, and device selection.
################################################################################

import torch
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _color_for_id(tid: int):
    return (37*tid % 256, 17*tid % 256, 93*tid % 256)

def expand_to_square(x1, y1, x2, y2, W, H, factor=1.3):
    """
    Expands a bounding box to a square region, centered on the original box.
    """
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    w, h = (x2-x1), (y2-y1)
    
    s = max(w, h) * factor 
    
    nx1 = int(max(0, math.floor(cx - 0.5*s)))
    ny1 = int(max(0, math.floor(cy - 0.5*s)))
    nx2 = int(min(W-1, math.ceil (cx + 0.5*s)))
    ny2 = int(min(H-1, math.ceil (cy + 0.5*s)))
    
    if nx2 <= nx1: nx2 = min(W-1, nx1+1)
    if ny2 <= ny1: ny2 = min(H-1, ny1+1)
    return nx1, ny1, nx2, ny2

def draw_tracks(frame: np.ndarray, tracks: np.ndarray, id_color_cache: dict):
    """Draw tracking boxes on frame."""
    for tb in tracks:
        x1, y1, x2, y2, tid, conf, cls, _ = tb
        x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
        color = id_color_cache.setdefault(tid, _color_for_id(tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, max(0, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame