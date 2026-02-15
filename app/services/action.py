################################################################################
# SlowFast Action Recognition logic.
################################################################################

import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import time
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from app.schemas import InferenceCfg, SlowFastInferCfg, ProcessingResult
from app.utils.video import _device, expand_to_square

def load_slowfast_classifier(labels, ckpt_path: str, device):
    torch.hub._validate_not_a_forked_repo = lambda a,b,c: True
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r101', pretrained=True)
    in_dim = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_dim, 128),
        nn.Dropout(p=0.3),
        nn.Linear(128, len(labels))
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model

class SlowFastPredictor:
    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = device or _device()

    @torch.no_grad()
    def predict_batch(self, slow_list: List[torch.Tensor], fast_list: List[torch.Tensor]) -> np.ndarray:
        if not slow_list: return np.empty((0,0))
        slow = torch.stack(slow_list).to(self.device)
        fast = torch.stack(fast_list).to(self.device)
        logits = self.model([slow, fast])
        return torch.softmax(logits, dim=1).cpu().numpy()

# --- Helper Functions for Data Prep ---
def load_tracks_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    for c in ("x1", "y1", "x2", "y2"):
        df[c] = df[c].astype(float)
    return df

def per_player_lookup(df: pd.DataFrame) -> Dict:
    look = {}
    for pid in df["id"].unique():
        sub = df[df["id"] == pid][["frame", "x1", "y1", "x2", "y2"]]
        look[int(pid)] = {int(r.frame): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) 
                          for r in sub.itertuples(index=False)}
    return look

def nearest_bbox(ff: int, lookup: Dict, span: int=3):
    if ff in lookup: return lookup[ff]
    for d in range(1, span+1):
        if ff-d in lookup: return lookup[ff-d]
        if ff+d in lookup: return lookup[ff+d]
    return None

def gather_dense_boxes(pid_lookup: Dict, t0: int, t1: int, span: int=3) -> Optional[List]:
    boxes = []
    for f in range(t0, t1+1):
        bb = nearest_bbox(f, pid_lookup, span=span)
        if bb is None: return None
        boxes.append(tuple(map(int, bb)))
    return boxes

def sample_indices_inference(L: int, slow_t: int, alpha: int):
    target_fast_frames = slow_t * alpha
    start_idx = (L - target_fast_frames) // 2
    idx_fast = list(range(start_idx, start_idx + target_fast_frames))
    idx_slow = idx_fast[::alpha]
    return idx_slow, idx_fast

def extract_clip_slowfast(frames_dict, segment, dense_boxes, sfcfg: SlowFastInferCfg):
    t0, t1 = segment
    L = t1 - t0 + 1
    if L <= 0 or len(dense_boxes) < L: return torch.empty(0), torch.empty(0)

    # EMA Smoothing
    smoothed = []
    prev = None
    for b in dense_boxes[:L]:
        arr = np.array(b, dtype=np.float32)
        prev = arr if prev is None else sfcfg.bbox_ema * prev + (1.0 - sfcfg.bbox_ema) * arr
        smoothed.append(tuple(prev.astype(int)))

    any_im = frames_dict.get(t0, None)
    if any_im is None: return torch.empty(0), torch.empty(0)
    H, W = any_im.shape[:2]

    # Union Box
    xs1 = [b[0] for b in smoothed]; ys1 = [b[1] for b in smoothed]
    xs2 = [b[2] for b in smoothed]; ys2 = [b[3] for b in smoothed]
    uni = (min(xs1), min(ys1), max(xs2), max(ys2))
    
    x1, y1, x2, y2 = expand_to_square(uni[0], uni[1], uni[2], uni[3], W, H, factor=sfcfg.bbox_margin)

    crops = []
    for k in range(L):
        fidx = t0 + k
        fr = frames_dict.get(fidx)
        if fr is None: fr = crops[-1] if len(crops) else any_im
        crop = fr[y1:y2, x1:x2]
        if crop.size == 0: crop = fr
        crop = cv2.resize(crop, (sfcfg.side, sfcfg.side), interpolation=cv2.INTER_AREA)
        crops.append(crop)

    idx_slow, idx_fast = sample_indices_inference(len(crops), sfcfg.slow_t, sfcfg.alpha)

    def _pack(idx):
        if not idx: return torch.empty(0)
        arr = np.stack([cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB) for i in idx]).astype(np.float32)/255.0
        arr = (arr - np.array(sfcfg.mean)) / np.array(sfcfg.std)
        return torch.from_numpy(np.transpose(arr, (3,0,1,2))).float()

    return _pack(idx_slow), _pack(idx_fast)

def run_action_recognition(video_path: str, tracks_csv: str, contact_frames: List[int], output_dir: str, 
                           model, cfg: InferenceCfg):
    import json
    start_time = time.time()
    try:
        if not contact_frames:
            return ProcessingResult(True, {'events': []}, processing_time=0.0)

        df = load_tracks_df(tracks_csv)
        by_pid = per_player_lookup(df)
        player_ids = sorted(by_pid.keys())

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prefetch needed frames
        needed = set()
        for c in contact_frames:
            t0 = max(0, c - cfg.n_before_after)
            t1 = min(total_frames - 1, c + cfg.n_before_after)
            needed.update(range(t0, t1+1))
        
        frame_cache = {}
        for fidx in sorted(needed):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            if ok: frame_cache[fidx] = fr
        cap.release()

        predictor = SlowFastPredictor(model)
        slow_batch, fast_batch, metas = [], [], []

        for c in contact_frames:
            t0, t1 = c - cfg.n_before_after, c + cfg.n_before_after
            if t0 < 0 or t1 >= total_frames: continue
            
            frames_dict = {f: frame_cache.get(f) for f in range(t0, t1+1)}
            
            for pid in player_ids:
                dense = gather_dense_boxes(by_pid.get(pid, {}), t0, t1, span=cfg.search_span)
                if dense is None: continue
                
                slow_t, fast_t = extract_clip_slowfast(frames_dict, (t0, t1), dense, cfg.sf)
                if slow_t.numel() == 0: continue
                
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
        
        events.sort(key=lambda e: (e["t0"], e["track_id"]))
        events_json_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_events.json")
        result_data = {"events": events, "contacts": contact_frames}
        
        with open(events_json_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        return ProcessingResult(True, {'events': events, 'events_json': events_json_path}, 
                              processing_time=time.time() - start_time)
    except Exception as e:
        return ProcessingResult(False, None, str(e))