###############################################################################
# YOLO + StrongSort Logic.
###############################################################################

import cv2
import time
import csv
import numpy as np
import os
from tqdm import tqdm
from app.schemas import YoloCfg, ProcessingResult
from app.utils.video import draw_tracks

def yolo_detect(yolo, frame, cfg: YoloCfg):
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
    return np.concatenate([boxes, confs[:, None], clss[:, None]], axis=1)

def run_strongsort(video_path: str, output_dir: str, yolo_cfg: YoloCfg, model_manager):
    start_time = time.time()
    try:
        video_name = os.path.basename(video_path).split(".")[0]
        tracks_video_path = os.path.join(output_dir, f"{video_name}_tracks.mp4")
        tracks_csv_path = os.path.join(output_dir, f"{video_name}_tracks.csv")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        
        out = cv2.VideoWriter(tracks_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        csv_rows = []
        id_color_cache = {}
        frame_idx = 0
        
        # Access shared models thread-safely
        with model_manager.yolo_lock:
            yolo = model_manager.yolo_model
            tracker = model_manager.strongsort_tracker
            
            pbar = tqdm(total=total, unit="frame", desc="Tracking")
            
            while True:
                ok, frame = cap.read()
                if not ok: break

                dets = yolo_detect(yolo, frame, yolo_cfg)
                
                if dets is None:
                    # No detections, just write frame
                    out.write(frame)
                else:
                    tracks = tracker.update(dets, frame)
                    if tracks is not None and len(tracks):
                        for x1, y1, x2, y2, tid, conf, cls, _ in tracks:
                            csv_rows.append((frame_idx, int(tid), int(x1), int(y1), 
                                           int(x2), int(y2), float(conf), int(cls)))
                        frame = draw_tracks(frame, tracks, id_color_cache)
                    out.write(frame)

                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            # Reset tracker for next video if needed (StrongSort typically handles this, but good practice)
            # tracker.reset() 

        cap.release()
        out.release()
        
        # Write CSV
        with open(tracks_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "id", "x1", "y1", "x2", "y2", "conf", "cls"])
            w.writerows(csv_rows)
            
        return ProcessingResult(
            True, 
            {'tracks_csv': tracks_csv_path, 'tracks_video': tracks_video_path}, 
            processing_time=time.time() - start_time
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ProcessingResult(False, None, str(e))