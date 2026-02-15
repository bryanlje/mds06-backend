import os
import sys
import csv
import json
import cv2
import math
import time
import tempfile
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence, DefaultDict
from collections import defaultdict
from queue import Queue
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ultralytics import YOLO
from boxmot import StrongSort
from google.cloud.storage import Client
from flask import Flask, request, jsonify, send_file
import ffmpeg
from google.cloud import storage as gcs
import logging
import re

logger = logging.getLogger(__name__)

# Define the base directories (as seen in your old code)
OUTPUTS_DIR = Path("/tmp/outputs")
# Ensure the directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def _gcs():
    """Initializes the GCS Client."""
    # Add retry/timeout configuration for robustness
    return gcs.Client()

def upload_blob(local_path: Path, bucket_name: str, blob_name: str):
    """Uploads a file and returns its GCS URI."""
    try:
        blob = _gcs().bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info("Uploaded %s -> %s", local_path, gcs_uri)
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {bucket_name}/{blob_name}: {e}")
        return None

# ============================================
# GLOBAL MODEL INSTANCES
# ============================================
yolo_model = None
strongsort_tracker = None
tracknet_model = None
inpaintnet_model = None
contact_detection_model = None
slowfast_model = None

# Thread locks
tracknet_lock = threading.Lock()
yolo_lock = threading.Lock()
contact_lock = threading.Lock()
slowfast_lock = threading.Lock()

# ============================================
# DATA CLASSES
# ============================================

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
class StrongSortCfg:
    max_age: Optional[int] = 30
    n_init: Optional[int] = 12
    max_iou_dist: Optional[float] = 1.0
    max_dist: Optional[float] = 1.0
    nn_budget: Optional[int] = 240
    half: bool = True
    det_thresh: float = 0.4

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
    mean: Tuple[float,float,float] = (0.45, 0.45, 0.45)
    std: Tuple[float,float,float] = (0.225, 0.225, 0.225)
    bbox_margin: float = 1.3
    bbox_ema: float = 0.8

@dataclass
class InferenceCfg:
    labels: List[str]
    sf: SlowFastInferCfg
    n_before_after: int = 18
    search_span: int = 3

@dataclass
class TrainCfg:
    labels = [
        "block", "clear", "cross_net",
        "drive", "drop", "jump_smash",
        "lift", "push", "serve",
        "smash", "straight_net", "tap",
        "negative"
    ]

# ============================================
# UTILITY FUNCTIONS
# ============================================

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _color_for_id(tid: int):
    return (37*tid % 256, 17*tid % 256, 93*tid % 256)

def _linspace_idx(a: int, b: int, n: int) -> List[int]:
    if n <= 1 or b <= a:
        return [a] * max(n, 1)
    return np.round(np.linspace(a, b, n)).astype(int).tolist()

def expand_to_square(x1, y1, x2, y2, W, H, factor=1.3):
    """
    Expands a bounding box to a square region, centered on the original box.
    """
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    w, h = (x2-x1), (y2-y1)
    
    # Use max dimension to create a square, padded by factor
    s = max(w, h) * factor 
    
    nx1 = int(max(0, math.floor(cx - 0.5*s)))
    ny1 = int(max(0, math.floor(cy - 0.5*s)))
    nx2 = int(min(W-1, math.ceil (cx + 0.5*s)))
    ny2 = int(min(H-1, math.ceil (cy + 0.5*s)))
    
    # Ensure box is valid
    if nx2 <= nx1: nx2 = min(W-1, nx1+1)
    if ny2 <= ny1: ny2 = min(H-1, ny1+1)
    return nx1,ny1,nx2,ny2

def setup_tracknet_repo():
    """Clone TrackNetV3 repository if not exists."""
    tracknet_dir = Path("./TrackNetV3")
    if not tracknet_dir.exists():
        print("Cloning TrackNetV3 repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/qaz812345/TrackNetV3.git"],
            check=True
        )
        print("✓ TrackNetV3 repository cloned")
    
    if str(tracknet_dir) not in sys.path:
        sys.path.insert(0, str(tracknet_dir))
    
    return tracknet_dir

def read_tracknet_csv(csv_path: str) -> Dict:
    """Read TrackNet output CSV."""
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_dict['Frame'].append(int(row['Frame']))
            pred_dict['X'].append(int(row['X']))
            pred_dict['Y'].append(int(row['Y']))
            pred_dict['Visibility'].append(int(row['Visibility']))
    
    return pred_dict

# ============================================
# YOLO & TRACKING FUNCTIONS
# ============================================

def yolo_detect(yolo: YOLO, frame: np.ndarray, cfg: YoloCfg):
    """Run YOLO detection on frame."""
    r = yolo.predict(
        frame, conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
        max_det=cfg.max_det, classes=cfg.classes,
        agnostic_nms=cfg.agnostic_nms, verbose=cfg.verbose
    )[0]
    
    if r.boxes is None or r.boxes.xyxy.numel() == 0:
        return None
    
    boxes = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy()
    clss = r.boxes.cls.detach().cpu().numpy()
    dets = np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1)
    
    return dets

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

def export_tracks_csv(csv_path: str, rows: List[Tuple]):
    """Export tracks to CSV."""
    if not csv_path:
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf", "cls"])
        w.writerows(rows)

def track_video(src_path: str, dst_path: str, yolo: YOLO, tracker: StrongSort,
                yolo_cfg: YoloCfg, export_csv: Optional[str] = None, show_pbar: bool = True):
    """Track players in video."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    print(f"🎥 Video: FPS={fps:.2f}, Frames={total}")
    
    pbar = tqdm(total=total, unit="frame", disable=not show_pbar)
    csv_rows = []
    id_color_cache = {}
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = yolo_detect(yolo, frame, yolo_cfg)
        if dets is None:
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            continue

        tracks = tracker.update(dets, frame)
        if tracks is not None and len(tracks):
            for x1, y1, x2, y2, tid, conf, cls, _ in tracks:
                csv_rows.append((frame_idx, int(tid), int(x1), int(y1), 
                               int(x2), int(y2), float(conf), int(cls)))
            frame = draw_tracks(frame, tracks, id_color_cache)

        cv2.putText(frame, f"Frame {frame_idx}", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"✓ Saved: {dst_path} | {frame_idx} frames | {time.time()-t0:.1f}s")
    
    if export_csv:
        export_tracks_csv(export_csv, csv_rows)
        print(f"✓ Saved CSV: {export_csv}")

# ============================================
# CONTACT DETECTION
# ============================================

class ContactDetectionCNN(nn.Module):
    """1D CNN for contact detection."""
    def __init__(self, input_features: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, hidden_dim, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv4 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = x2 + x1
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = x4 + x3
        x_pooled = self.pool(x4).squeeze(-1)
        return self.fc(x_pooled)

class ContactDetector:
    """Contact detection inference."""
    def __init__(self, model: nn.Module, cfg: ContactConfig, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg

    def predict_csv(self, csv_path: str, confidence_threshold: float = 0.5, 
                   nms_window: int = 10) -> List[int]:
        frames, vis, xs, ys = self._load_csv(csv_path)
        features = self._extract_features(xs, ys, vis)
        probabilities = self._sliding_window_inference(features)
        half_window = self.cfg.window_size // 2
        valid_frames = frames[half_window:-half_window]
        return self._detect_peaks(valid_frames, probabilities, confidence_threshold, nms_window)

    def _load_csv(self, path: str):
        frames, vis, xs, ys = [], [], [], []
        with open(path, 'r') as f:
            for r in csv.DictReader(f):
                frames.append(int(r["Frame"]))
                vis.append(int(r["Visibility"]))
                xs.append(float(r["X"]))
                ys.append(float(r["Y"]))
        
        frames, vis = np.array(frames), np.array(vis)
        xs, ys = np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
        idx = np.argsort(frames)
        return frames[idx], vis[idx], xs[idx], ys[idx]

    def _extract_features(self, xs, ys, vis):
        xs_filled, ys_filled = xs.copy(), ys.copy()
        valid = (vis == 1) & ~((xs == 0) & (ys == 0))
        
        for i in range(1, len(xs)):
            if not valid[i]:
                xs_filled[i], ys_filled[i] = xs_filled[i-1], ys_filled[i-1]

        def smooth(arr, window):
            if window < 2:
                return arr
            if window % 2 == 0:
                window += 1
            pad = window // 2
            padded = np.pad(arr, pad, mode='edge')
            kernel = np.ones(window) / window
            return np.convolve(padded, kernel, mode='valid').astype(np.float32)

        xs_s, ys_s = smooth(xs_filled, self.cfg.smooth_window), smooth(ys_filled, self.cfg.smooth_window)
        features = [xs_s, ys_s]

        vx, vy = np.diff(xs_s, prepend=xs_s[0]), np.diff(ys_s, prepend=ys_s[0])
        speed = np.sqrt(vx**2 + vy**2)
        features.extend([vx, vy, speed])

        ax, ay = np.diff(vx, prepend=vx[0]), np.diff(vy, prepend=vy[0])
        features.extend([ax, ay])

        direction = np.arctan2(vy, vx)
        direction_change = np.diff(direction, prepend=direction[0])
        features.extend([direction, direction_change])

        return np.stack(features, axis=1)

    def _sliding_window_inference(self, features):
        N, half_window = len(features), self.cfg.window_size // 2
        probabilities = []

        with torch.no_grad():
            for i in range(half_window, N - half_window):
                window = features[i - half_window:i + half_window + 1]
                window_tensor = torch.tensor(window, dtype=torch.float32).T.unsqueeze(0).to(self.device)
                output = self.model(window_tensor)
                prob = F.softmax(output, dim=1)[0, 1].item()
                probabilities.append(prob)

        return np.array(probabilities)

    def _detect_peaks(self, frames, probabilities, threshold, nms_window):
        candidates = [(frames[i], prob) for i, prob in enumerate(probabilities) if prob >= threshold]
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for frame, prob in candidates:
            if not any(abs(frame - sel) < nms_window for sel in selected):
                selected.append(frame)
        
        return sorted(selected)

# ============================================
# SLOWFAST FUNCTIONS
# ============================================

def load_tracks_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    for c in ("x1", "y1", "x2", "y2"):
        df[c] = df[c].astype(float)
    return df

def per_player_lookup(df: pd.DataFrame) -> Dict[int, Dict[int, Tuple[float,float,float,float]]]:
    look = {}
    for pid in df["id"].unique():
        sub = df[df["id"] == pid][["frame", "x1", "y1", "x2", "y2"]]
        look[int(pid)] = {int(r.frame): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) 
                         for r in sub.itertuples(index=False)}
    return look

def nearest_bbox(ff: int, lookup: Dict[int, Tuple[float,float,float,float]], span: int=3):
    if ff in lookup:
        return lookup[ff]
    for d in range(1, span+1):
        if ff-d in lookup:
            return lookup[ff-d]
        if ff+d in lookup:
            return lookup[ff+d]
    return None

def gather_dense_boxes(pid_lookup: Dict, t0: int, t1: int, span: int=3) -> Optional[List[Tuple[int,int,int,int]]]:
    boxes = []
    for f in range(t0, t1+1):
        bb = nearest_bbox(f, pid_lookup, span=span)
        if bb is None:
            return None
        x1, y1, x2, y2 = bb
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

def sample_indices_inference(L: int, slow_t: int, alpha: int):
    """
    For inference: Grabs the center contiguous block of frames.
    L = Total frames available (e.g., 39)
    """
    # Calculate how many fast frames the model needs
    target_fast_frames = slow_t * alpha  # e.g., 8 * 4 = 32

    # Find the start index for the center block
    # e.g., (39 - 32) // 2 = 7 // 2 = 3
    start_idx = (L - target_fast_frames) // 2

    # Create the contiguous list of fast indices
    # e.g., list(range(3, 3 + 32)) -> [3, 4, ..., 34]
    idx_fast = list(range(start_idx, start_idx + target_fast_frames))

    # Sample the slow path from the fast path
    # e.g., [3, 7, 11, 15, 19, 23, 27, 31]
    idx_slow = idx_fast[::alpha]

    return idx_slow, idx_fast

def extract_clip_slowfast(frames_dict: Dict[int, np.ndarray],
                          segment: Tuple[int,int],
                          dense_boxes: List[Tuple[int,int,int,int]],
                          sfcfg: SlowFastInferCfg):
    """
    frames_dict: {frame_idx -> BGR frame}
    segment: inclusive [t0, t1]
    dense_boxes: per-frame (x1,y1,x2,y2) aligned to t0..t1 (same length as segment)
    """
    t0, t1 = segment
    L = t1 - t0 + 1
    if L <= 0 or len(dense_boxes) < L:
        return torch.empty(0), torch.empty(0)

    # Smooth bbox sequence (EMA)
    smoothed = []
    prev = None
    for b in dense_boxes[:L]:
        arr = np.array(b, dtype=np.float32)
        prev = arr if prev is None else sfcfg.bbox_ema * prev + (1.0 - sfcfg.bbox_ema) * arr
        smoothed.append(tuple(prev.astype(int)))

    # Collect crops
    # infer frame shape from any available frame
    any_im = frames_dict.get(t0, None)
    if any_im is None:
        return torch.empty(0), torch.empty(0)
    H, W = any_im.shape[:2]

    # 1. Get union of all smoothed boxes
    xs1 = [b[0] for b in smoothed]; ys1 = [b[1] for b in smoothed]
    xs2 = [b[2] for b in smoothed]; ys2 = [b[3] for b in smoothed]
    uni = (min(xs1), min(ys1), max(xs2), max(ys2))

    # 2. Expand union box ONCE
    # (Using the new unified function)
    x1, y1, x2, y2 = expand_to_square(uni[0], uni[1], uni[2], uni[3], W, H, factor=sfcfg.bbox_margin)
    crop_box = (x1, y1, x2, y2)

    crops = []
    for k in range(L):
        fidx = t0 + k
        fr = frames_dict.get(fidx)
        if fr is None:
            # pad by repeating last valid frame
            fr = crops[-1] if len(crops) else any_im

        crop = fr[y1:y2, x1:x2]

        if crop.size == 0:
            crop = fr
        crop = cv2.resize(crop, (sfcfg.side, sfcfg.side), interpolation=cv2.INTER_AREA)
        crops.append(crop)

    idx_slow, idx_fast = sample_indices_inference(len(crops), sfcfg.slow_t, sfcfg.alpha)

    def _pack(frames_bgr_idx):
        if not frames_bgr_idx:
            return torch.empty(0)
        arr = np.stack([cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in frames_bgr_idx]).astype(np.float32)/255.0
        arr = (arr - np.array(sfcfg.mean)) / np.array(sfcfg.std)
        # (T,H,W,C) -> (C,T,H,W)
        return torch.from_numpy(np.transpose(arr, (3,0,1,2))).float()

    slow = _pack(idx_slow)
    fast = _pack(idx_fast)
    return slow, fast

class SlowFastPredictor:
    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = device or _device()

    @torch.no_grad()
    def predict_batch(self, slow_list: List[torch.Tensor], fast_list: List[torch.Tensor]) -> np.ndarray:
        if not slow_list:
            return np.empty((0,0))
        slow = torch.stack(slow_list).to(self.device)
        fast = torch.stack(fast_list).to(self.device)
        logits = self.model([slow, fast])
        return torch.softmax(logits, dim=1).cpu().numpy()

def load_slowfast_classifier(cfg, ckpt_path: str, device: Optional[torch.device] = None):
    device = device or _device()
    print(f"🧠 Loading SlowFast from: {ckpt_path}")
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
    in_dim = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_dim, 128),
        nn.Dropout(p=0.3),
        nn.Linear(128, len(cfg.labels))
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    print(f"   ✓ SlowFast loaded on {device}")
    return model

def run_full_video_to_events(video_path: str, tracks_csv: str, contact_frames: List[int],
                             model, cfg: InferenceCfg) -> Dict:
    df = load_tracks_df(tracks_csv)
    by_pid = per_player_lookup(df)
    player_ids = sorted(by_pid.keys())

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames: {total_frames}")

    frame_cache = {}
    def get_frame(fidx: int):
        if fidx in frame_cache:
            return frame_cache[fidx]
        if fidx < 0 or fidx >= total_frames:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, fr = cap.read()
        if ok:
            frame_cache[fidx] = fr
        return fr if ok else None

    needed = set()
    for c in contact_frames:
        t0 = max(0, c - cfg.n_before_after)
        t1 = min(total_frames - 1, c + cfg.n_before_after)
        needed.update(range(t0, t1+1))
    
    for fidx in tqdm(sorted(needed), desc="Prefetch frames"):
        _ = get_frame(fidx)

    predictor = SlowFastPredictor(model, device=_device())
    slow_batch, fast_batch, metas = [], [], []

    for c in tqdm(contact_frames, desc="Process contacts"):
        t0, t1 = c - cfg.n_before_after, c + cfg.n_before_after
        if t0 < 0 or t1 >= total_frames:
            continue

        frames_dict = {f: frame_cache.get(f) for f in range(t0, t1+1)}
        for pid in player_ids:
            dense = gather_dense_boxes(by_pid[pid], t0, t1, span=cfg.search_span)
            if dense is None:
                continue

            slow_t, fast_t = extract_clip_slowfast(frames_dict, (t0, t1), dense, cfg.sf)
            if slow_t.numel() == 0 or fast_t.numel() == 0:
                continue

            slow_batch.append(slow_t)
            fast_batch.append(fast_t)
            metas.append((pid, t0, t1))

    events = []
    if slow_batch:
        probs = predictor.predict_batch(slow_batch, fast_batch)
        for (pid, t0, t1), p in zip(metas, probs):
            k = int(np.argmax(p))
            events.append({
                "track_id": int(pid),
                "t0": int(t0),
                "t1": int(t1),
                "label": cfg.labels[k],
                "p": float(p[k]),
            })

    cap.release()
    events.sort(key=lambda e: (e["t0"], e["track_id"]))
    return {"events": events, "contacts": [int(c) for c in contact_frames]}

# ============================================
# VIDEO OVERLAY
# ============================================

def load_events_and_contacts(events_json_path: str):
    if not os.path.exists(events_json_path):
        return [], []
    with open(events_json_path, "r") as f:
        data = json.load(f)
    return data.get("events", []), data.get("contacts", [])

def build_event_map_by_frame(events: List[Dict]) -> Dict[int, Dict[int, Dict]]:
    by_frame = defaultdict(dict)
    for e in events:
        for fr in range(int(e["t0"]), int(e["t1"]) + 1):
            by_frame[fr][int(e["track_id"])] = e
    return by_frame

def render_full_video_overlay(video_frames: List[np.ndarray], tracks_csv: str, shuttle_csv: str,
                              events_json: str, out_path: str, fps: float, 
                              show_ids: bool = True, label_bg_alpha: float = 0.4):
    boxes_by_frame = defaultdict(list)
    with open(tracks_csv, "r") as f:
        for r in csv.DictReader(f):
            boxes_by_frame[int(r["frame"])].append(
                (int(r["id"]), (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))))

    shuttle_coordinates = defaultdict(lambda: [0, 0])
    with open(shuttle_csv, "r") as f2:
        for r in csv.DictReader(f2):
            shuttle_coordinates[int(r["Frame"])] = [int(r["X"]), int(r["Y"])]

    events, contacts = load_events_and_contacts(events_json)
    event_by_frame = build_event_map_by_frame(events)

    if not video_frames:
        print("⚠ No frames to render")
        return
    
    H, W, _ = video_frames[0].shape
    scale = max(H, W) / 1080.0
    BOX_TH = max(3, int(6 * scale))
    OUTLINE_TH = BOX_TH + max(2, int(2 * scale))
    FONT_SCALE = max(0.8, 0.9 * scale)
    TXT_TH = max(2, int(3 * scale))
    STROKE_TH = TXT_TH + max(1, int(2 * scale))
    PAD = max(6, int(8 * scale))
    BG_ALPHA = max(label_bg_alpha, 0.65)

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{W}x{H}', r=fps)
        .output(
            out_path, 
            vcodec='libx264',    # The video codec
            pix_fmt='yuv420p'    # The *most important* part for web
        ) 
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        for i, frame in tqdm(enumerate(video_frames), desc="Render overlay"):
            overlay = frame.copy()

            # Shuttlecock
            cv2.circle(overlay, tuple(shuttle_coordinates[i]), 10, _color_for_id(5), 5)

            # Frame number
            cv2.putText(overlay, f"Frame {i}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 200), 4)

            # Contact indicator
            if i in contacts or i-1 in contacts or i+1 in contacts:
                cv2.putText(overlay, "Contact", (50, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE*1.2, (0, 0, 255), 4)

            # Player boxes
            for tid, (x1, y1, x2, y2) in boxes_by_frame.get(i, []):
                color = _color_for_id(tid)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), OUTLINE_TH, cv2.LINE_AA)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, BOX_TH, cv2.LINE_AA)

                ev = event_by_frame.get(i, {}).get(tid)
                if ev and ev['label'] != "negative":
                    text = f"{ev['label']} {ev['p']:.2f}"
                elif show_ids:
                    text = f"ID {tid}"
                else:
                    text = None

                if text:
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TXT_TH)
                    tx, ty = x1 + BOX_TH, y1 - PAD
                    if ty - th - PAD < 0:
                        ty = y1 + th + PAD

                    bg = overlay.copy()
                    cv2.rectangle(bg, (tx - PAD, ty - th - int(1.2 * PAD)),
                                (tx + tw + PAD, ty + int(0.6 * PAD)), color, -1)
                    overlay = cv2.addWeighted(bg, BG_ALPHA, overlay, 1 - BG_ALPHA, 0)

                    cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                              FONT_SCALE, (0, 0, 0), STROKE_TH, cv2.LINE_AA)
                    cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                              FONT_SCALE, (255, 255, 255), TXT_TH, cv2.LINE_AA)

            process.stdin.write(overlay.tobytes())

    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        print(f"✓ Video saved: {out_path}")

# ============================================
# CONCURRENT PIPELINE STEPS
# ============================================

def run_step_1_strongsort(video_path: str, output_dir: str, yolo_cfg: YoloCfg, result_queue: Queue):
    """Step 1: Player tracking."""
    try:
        start_time = time.time()
        print(f"[StrongSort] Starting...")
        
        video_name = Path(video_path).stem
        tracks_video_path = os.path.join(output_dir, f"{video_name}_tracks.mp4")
        tracks_csv_path = os.path.join(output_dir, f"{video_name}_tracks.csv")
        
        with yolo_lock:
            track_video(video_path, tracks_video_path, yolo_model, strongsort_tracker,
                       yolo_cfg, tracks_csv_path, True)
        
        result_queue.put(('strongsort', ProcessingResult(
            True, {'tracks_video': tracks_video_path, 'tracks_csv': tracks_csv_path},
            processing_time=time.time() - start_time
        )))
        print(f"[StrongSort] ✓ Done in {time.time()-start_time:.2f}s")
        
    except Exception as e:
        print(f"[StrongSort] ❌ Error: {e}")
        result_queue.put(('strongsort', ProcessingResult(False, None, str(e))))

def run_step_2_tracknet_subprocess(video_path: str, output_dir: str, tracknet_weights: str,
                                   inpaintnet_weights: str, batch_size: int, eval_mode: str,
                                   result_queue: Queue):
    """Step 2: Shuttlecock tracking via subprocess."""
    try:
        start_time = time.time()
        print(f"[TrackNet] Starting...")
        
        video_name = Path(video_path).stem
        cmd = [
            sys.executable, "-m", "TrackNetV3.predict",
            "--video_file", video_path,
            "--tracknet_file", tracknet_weights,
            "--inpaintnet_file", inpaintnet_weights,
            "--save_dir", output_dir,
            "--batch_size", str(batch_size),
            "--eval_mode", eval_mode,
            # "--output_video", "False"
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(f"[TrackNet] {line.strip()}")
        
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"TrackNet failed: {process.stderr.read()}")
        
        expected_csv = os.path.join(output_dir, f"{video_name}_ball.csv")
        if not os.path.exists(expected_csv):
            raise FileNotFoundError(f"CSV not found: {expected_csv}")
        
        pred_dict = read_tracknet_csv(expected_csv)
        
        result_queue.put(('tracknet', ProcessingResult(
            True, {'shuttle_csv': expected_csv, 'predictions': pred_dict},
            processing_time=time.time() - start_time
        )))
        print(f"[TrackNet] ✓ Done in {time.time()-start_time:.2f}s")
        
    except Exception as e:
        print(f"[TrackNet] ❌ Error: {e}")
        result_queue.put(('tracknet', ProcessingResult(False, None, str(e))))

def run_step_3_contact_detection(shuttle_csv: str, output_dir: str,
                                 confidence_threshold: float = 0.6, nms_window: int = 10):
    """Step 3: Contact detection."""
    try:
        start_time = time.time()
        print(f"[Contact] Starting...")
        
        if contact_detection_model is None:
            return ProcessingResult(False, None, "Contact model not loaded")
        
        with contact_lock:
            detector = ContactDetector(contact_detection_model, ContactConfig())
            contact_frames = detector.predict_csv(shuttle_csv, confidence_threshold, nms_window)
        
        print(f"[Contact] ✓ Found {len(contact_frames)} frames in {time.time()-start_time:.2f}s")
        return ProcessingResult(True, {'contact_frames': [int(f) for f in contact_frames]},
                              processing_time=time.time() - start_time)
        
    except Exception as e:
        print(f"[Contact] ❌ Error: {e}")
        return ProcessingResult(False, None, str(e))

def run_step_4_slowfast(video_path: str, tracks_csv: str, contact_frames: List[int], output_dir: str):
    """Step 4: Action recognition."""
    try:
        start_time = time.time()
        print(f"[SlowFast] Starting...")
        
        if slowfast_model is None:
            return ProcessingResult(False, None, "SlowFast not loaded")
        
        if not contact_frames:
            print(f"[SlowFast] ⊘ No contacts")
            return ProcessingResult(True, {'events': []}, 0.0)
        
        cfg = InferenceCfg(TrainCfg().labels, SlowFastInferCfg(), 18, 3)
        
        with slowfast_lock:
            result_data = run_full_video_to_events(video_path, tracks_csv, contact_frames,
                                                   slowfast_model, cfg)
        
        events_json = os.path.join(output_dir, f"{Path(video_path).stem}_events.json")
        with open(events_json, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"[SlowFast] ✓ {len(result_data['events'])} events in {time.time()-start_time:.2f}s")
        return ProcessingResult(True, {'events': result_data['events'], 'events_json': events_json},
                              processing_time=time.time() - start_time)
        
    except Exception as e:
        print(f"[SlowFast] ❌ Error: {e}")
        return ProcessingResult(False, None, str(e))

def run_step_5_overlay(video_path: str, tracks_csv: str, shuttle_csv: str, events_json: str,
                      contact_frames: List[int], output_dir: str):
    """Step 5: Render overlay."""
    try:
        start_time = time.time()
        print(f"[Overlay] Starting...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        cap.release()
        
        overlay_path = os.path.join(output_dir, f"{Path(video_path).stem}_overlay.mp4")
        render_full_video_overlay(all_frames, tracks_csv, shuttle_csv, events_json or '',
                                 overlay_path, fps, True, 0.4)
        
        print(f"[Overlay] ✓ Done in {time.time()-start_time:.2f}s")
        return ProcessingResult(True, {'overlay_video': overlay_path},
                              processing_time=time.time() - start_time)
        
    except Exception as e:
        print(f"[Overlay] ❌ Error: {e}")
        return ProcessingResult(False, None, str(e))

def process_video_concurrent(video_path: str, output_dir: str,
                            tracknet_weights: str = '/app/TrackNetV3/ckpts/TrackNet_best.pt',
                            inpaintnet_weights: str = '/app/TrackNetV3/ckpts/InpaintNet_best.pt',
                            batch_size: int = 16, eval_mode: str = 'weight',
                            yolo_cfg: Optional[YoloCfg] = None):
    """Main concurrent pipeline."""
    print("\n" + "="*70)
    print("🚀 CONCURRENT VIDEO PROCESSING")
    print("="*70 + "\n")
    
    pipeline_start = time.time()
    if yolo_cfg is None:
        yolo_cfg = YoloCfg()
    
    result_queue = Queue()
    
    # Parallel steps
    print("🔄 Steps 1 & 2 (parallel)...")
    step_1 = threading.Thread(target=run_step_1_strongsort,
                              args=(video_path, output_dir, yolo_cfg, result_queue),
                              name="StrongSort")
    step_2 = threading.Thread(target=run_step_2_tracknet_subprocess,
                              args=(video_path, output_dir, tracknet_weights, inpaintnet_weights,
                                   batch_size, eval_mode, result_queue),
                              name="TrackNet")
    
    parallel_start = time.time()
    step_1.start()
    step_2.start()
    step_1.join()
    step_2.join()
    parallel_time = time.time() - parallel_start
    
    print(f"\n✓ Parallel complete: {parallel_time:.2f}s\n")
    
    # Collect results
    results = {}
    while not result_queue.empty():
        name, result = result_queue.get()
        results[name] = result
    
    if not results.get('strongsort', ProcessingResult(False, None)).success:
        return {'success': False, 'error': results['strongsort'].error, 'stage': 'strongsort'}
    if not results.get('tracknet', ProcessingResult(False, None)).success:
        return {'success': False, 'error': results['tracknet'].error, 'stage': 'tracknet'}
    
    tracks_csv = results['strongsort'].data['tracks_csv']
    tracks_video = results['strongsort'].data['tracks_video']
    shuttle_csv = results['tracknet'].data['shuttle_csv']
    
    # Sequential steps
    print("💥 Step 3: Contact Detection")
    contact_result = run_step_3_contact_detection(shuttle_csv, output_dir, 0.6, 10)
    contact_frames = contact_result.data['contact_frames'] if contact_result.success else []
    
    print("\n🎬 Step 4: Action Recognition")
    slowfast_result = run_step_4_slowfast(video_path, tracks_csv, contact_frames, output_dir)
    events = slowfast_result.data['events'] if slowfast_result.success else []
    events_json = slowfast_result.data.get('events_json') if slowfast_result.success else None
    
    print("\n🎨 Step 5: Overlay")
    overlay_result = run_step_5_overlay(video_path, tracks_csv, shuttle_csv,
                                       events_json or '', contact_frames, output_dir)
    overlay_video = overlay_result.data['overlay_video'] if overlay_result.success else None
    
    total_time = time.time() - pipeline_start
    time_saved = (results['strongsort'].processing_time + 
                 results['tracknet'].processing_time) - parallel_time
    
    print("\n" + "="*70)
    print("✨ PIPELINE COMPLETE")
    print(f"Total: {total_time:.2f}s | Saved: {time_saved:.2f}s ({time_saved/total_time*100:.1f}%)")
    print("="*70 + "\n")
    
    return {
        'success': True,
        'tracks_csv': tracks_csv,
        'tracks_video': tracks_video,
        'shuttle_csv': shuttle_csv,
        'contact_frames': contact_frames,
        'events': events,
        'events_json': events_json,
        'overlay_video': overlay_video,
        'timing': {
            'strongsort': results['strongsort'].processing_time,
            'tracknet': results['tracknet'].processing_time,
            'parallel_time': parallel_time,
            'contact_detection': contact_result.processing_time if contact_result.success else 0,
            'action_recognition': slowfast_result.processing_time if slowfast_result.success else 0,
            'overlay_rendering': overlay_result.processing_time if overlay_result.success else 0,
            'total_time': total_time,
            'time_saved': time_saved
        }
    }

# ============================================
# MODEL LOADING
# ============================================

def load_models():
    """Load all models at container startup."""
    global yolo_model, strongsort_tracker, tracknet_model, inpaintnet_model
    global contact_detection_model, slowfast_model
    
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60 + "\n")
    
    TRACKNET_PATH = os.getenv('TRACKNET_WEIGHTS_PATH', '/app/TrackNetV3/ckpts/TrackNet_best.pt')
    INPAINTNET_PATH = os.getenv('INPAINTNET_WEIGHTS_PATH', '/app/TrackNetV3/ckpts/InpaintNet_best.pt')
    CONTACT_PATH = os.getenv('CONTACT_WEIGHTS_PATH', '/app/models/contact_model.pth')
    SLOWFAST_PATH = os.getenv('SLOWFAST_WEIGHTS_PATH', '/app/models/slowfast_model.pt')
    YOLO_PATH = os.getenv('YOLO_WEIGHTS_PATH', '/app/models/yolo_weights.pt')
    REID_PATH = os.getenv('REID_WEIGHTS_PATH', '/app/models/osnet_x1_0_badminton.pt')
    
    USE_INPAINTNET = os.getenv('USE_INPAINTNET', 'false').lower() == 'true'
    USE_CONTACT = os.getenv('USE_CONTACT_DETECTION', 'true').lower() == 'true'
    USE_SLOWFAST = os.getenv('USE_SLOWFAST', 'true').lower() == 'true'
    USE_YOLO = os.getenv('USE_YOLO', 'true').lower() == 'true'
    
    device = _device()
    print(f"Device: {device}\n")
    
    # Verify files
    required = [TRACKNET_PATH]
    if USE_INPAINTNET:
        required.append(INPAINTNET_PATH)
    if USE_CONTACT:
        required.append(CONTACT_PATH)
    if USE_SLOWFAST:
        required.append(SLOWFAST_PATH)
    if USE_YOLO:
        required.extend([YOLO_PATH, REID_PATH])
    
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {path}")
        print(f"✓ {path}")
    
    # Setup TrackNet repo (already exists in container)
    setup_tracknet_repo()
    
    # Load models (simplified - TrackNet via subprocess, so we don't load it here)
    print("\n📦 Loading models...")
    
    if USE_YOLO:
        print("  → YOLO")
        yolo_model = YOLO(YOLO_PATH)
        print("  → StrongSort")
        strongsort_tracker = StrongSort(
            reid_weights=Path(REID_PATH),
            device=0 if torch.cuda.is_available() else 'cpu',
            half=(StrongSortCfg.half and torch.cuda.is_available()),
            # BaseTracker parameters
            det_thresh=StrongSortCfg.det_thresh,
            max_age=StrongSortCfg.max_age,
            n_init=StrongSortCfg.n_init,
            max_iou_dist=StrongSortCfg.max_iou_dist,
            max_cos_dist=StrongSortCfg.max_dist,
            nn_budget=StrongSortCfg.nn_budget
        )
        print("    ✓ YOLO + StrongSort")
    
    if USE_CONTACT:
        print("  → Contact Detection")
        contact_detection_model = ContactDetectionCNN(9, 64)
        contact_detection_model.load_state_dict(torch.load(CONTACT_PATH, map_location=device))
        contact_detection_model.to(device).eval()
        print("    ✓ Contact Detection")
    
    if USE_SLOWFAST:
        print("  → SlowFast")
        slowfast_model = load_slowfast_classifier(TrainCfg(), SLOWFAST_PATH, device)
        print("    ✓ SlowFast")
    
    print("\n" + "="*60)
    print("✅ ALL MODELS LOADED")
    print("="*60 + "\n")

# ============================================
# FLASK APP
# ============================================

print("\n🚀 Container starting...")
load_models()
print("✅ Ready!\n")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check."""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "yolo": yolo_model is not None,
            "strongsort": strongsort_tracker is not None,
            "contact": contact_detection_model is not None,
            "slowfast": slowfast_model is not None
        },
        "device": str(_device()),
        "threading": True
    })

def _parse_gcs_uri(gcs_uri: str) -> (str, str):
    """Parses a GCS URI into (bucket, blob_name)."""
    # Use a regex to be more robust
    match = re.match(r"gs://([^/]+)/(.+)", gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
    bucket_name, blob_name = match.groups()
    return bucket_name, blob_name

# --- New GCS Download Function ---
def download_blob(gcs_uri: str, local_path: str):
    """Downloads a blob from GCS to a local file."""
    try:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        blob = _gcs().bucket(bucket_name).blob(blob_name)
        
        if not blob.exists():
            logger.error(f"Blob not found at GCS URI: {gcs_uri}")
            raise FileNotFoundError(f"Blob not found: {gcs_uri}")

        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {gcs_uri} -> {local_path}")
    except Exception as e:
        logger.exception(f"Failed to download {gcs_uri}")
        raise e
    

@app.route('/process', methods=['POST'])
def process_video_endpoint():
    """Main concurrent processing endpoint."""
    input_path = None # Will store the path to the local temp file
    
    try:
        # --- Get input from EITHER file upload or GCS URI ---
        video_file = request.files.get('video')
        gcs_uri = request.form.get('gcs_uri')

        if not video_file and not gcs_uri:
            return jsonify({"detail": "Provide either a 'video' file upload or a 'gcs_uri' form field."}), 400
        if video_file and gcs_uri:
            return jsonify({"detail": "Provide only one of 'video' or 'gcs_uri', not both."}), 400    
        
        # --- Get parameters from form-data ---
        batch_size = int(request.form.get('batch_size', 16))
        eval_mode = request.form.get('eval_mode', 'weight')
        output_bucket = request.form.get('output_bucket') 
        
        # === NEW: Get user and job info ===
        user_id = request.form.get('user_id', 'unknown_user')
        job_id = request.form.get('job_id', 'unknown_job')
        # ==================================

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            input_path = tmp.name

        input_source_name = None

        try:
            # === FIX 1 (Part B): Now, save or download to the path ===
            if video_file:
                video_file.save(input_path)
                input_source_name = video_file.filename # === FIX 2 (Part B) ===
                print(f"\n📹 Saved uploaded file: {input_source_name} -> {input_path}")
            elif gcs_uri:
                input_source_name = gcs_uri # === FIX 2 (Part B) ===
                print(f"\n📹 Downloading GCS file: {input_source_name} -> {input_path}")
                download_blob(gcs_uri, input_path)
        except Exception as e:
            # Handle a failed download/save
            logger.exception(f"Failed to get input video: {e}")
            return jsonify({"detail": f"Failed to get input video: {e}"}), 500
        
        output_dir = tempfile.mkdtemp(dir=OUTPUTS_DIR)
        
        print(f"\n🚀 Processing: {input_source_name} -> {output_dir}")
        
        result = process_video_concurrent(
            input_path, output_dir,
            batch_size=batch_size,
            eval_mode=eval_mode
        )
        
        if not result['success']:
            error_message = result.get('error', 'Unknown processing failure')
            return jsonify({"detail": f"Pipeline failed: {error_message}"}), 500
        
        if not result['success']:
            error_message = result.get('error', 'Unknown processing failure')
            return jsonify({"detail": f"Pipeline failed: {error_message}"}), 500
        
        # --- GCS Upload Logic ---
        # This 'gcs_results' dict will be our 'outputs'
        gcs_results = {}
        if output_bucket:
            logger.info(f"Uploading results to GCS bucket: {output_bucket}")
            # Loop over the files generated by your process
            for key, local_path_str in result.items():
                # Only upload keys that are files and exist
                if key in ['tracks_csv', 'shuttle_csv', 'overlay_video'] and local_path_str and os.path.exists(local_path_str):
                    local_path = Path(local_path_str)
                    # Use job_id or user_id in the path for better organization
                    blob_name = f"outputs/{user_id}/{job_id}/{local_path.name}"
                    gcs_uri = upload_blob(local_path, output_bucket, blob_name)
                    gcs_results[key] = gcs_uri # This is our 'uploads' dict
        
        # === NEW: Build the final output ===
        # Get the events list from your processing result
        events_list = result.get('events', [])

        # Build the dictionary to match your old format
        final_output = {
            "job": {
                "user_id": user_id,
                "job_id": job_id
            },
            "outputs": gcs_results if output_bucket else "Not requested",
            "events": events_list
        }
        # ===================================
        
        return jsonify(final_output)
    
    except Exception as e:
        logger.exception("Fatal endpoint error")
        return jsonify({"detail": str(e), "trace": "Pipeline failed"}), 500
    
    finally:
        # === FIX 3: Clean up the temp file, no matter what happens ===
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
                print(f"🧹 Cleaned up temp input file: {input_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temp file {input_path}: {e}")

@app.route('/download/<file_type>', methods=['GET'])
def download_file(file_type):
    """Download generated files."""
    try:
        path = request.args.get('path')
        if not path or not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404
        return send_file(path, as_attachment=True, download_name=os.path.basename(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)