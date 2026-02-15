################################################################################
# Contact Detection Model Definition and Logic.
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from typing import List
from app.schemas import ContactConfig

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
    def __init__(self, model: nn.Module, cfg: ContactConfig, device: str):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg

    def predict_csv(self, csv_path: str, confidence_threshold: float = 0.5, nms_window: int = 10) -> List[int]:
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
            if window < 2: return arr
            if window % 2 == 0: window += 1
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
        if not candidates: return []
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = []
        for frame, prob in candidates:
            if not any(abs(frame - sel) < nms_window for sel in selected):
                selected.append(frame)
        return sorted(selected)

def run_contact_detection(shuttle_csv: str, model_instance: nn.Module, device: str, 
                          confidence_threshold: float = 0.6, nms_window: int = 10):
    from app.Schemas import ProcessingResult
    import time
    try:
        start_time = time.time()
        detector = ContactDetector(model_instance, ContactConfig(), device)
        contact_frames = detector.predict_csv(shuttle_csv, confidence_threshold, nms_window)
        return ProcessingResult(True, {'contact_frames': [int(f) for f in contact_frames]}, 
                              processing_time=time.time() - start_time)
    except Exception as e:
        return ProcessingResult(False, None, str(e))