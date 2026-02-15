###############################################################################
# Overlay rendering using ffmpeg python.
###############################################################################

import cv2
import csv
import json
import ffmpeg
import time
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from app.utils.video import _color_for_id
from app.schemas import ProcessingResult

def load_events_and_contacts(events_json_path: str):
    if not os.path.exists(events_json_path): return [], []
    with open(events_json_path, "r") as f:
        data = json.load(f)
    return data.get("events", []), data.get("contacts", [])

def run_overlay(video_path: str, tracks_csv: str, shuttle_csv: str, events_json: str, 
                contact_frames: list, output_dir: str):
    start_time = time.time()
    try:
        video_name = os.path.basename(video_path).split(".")[0]
        out_path = os.path.join(output_dir, f"{video_name}_overlay.mp4")

        # Load Data
        boxes_by_frame = defaultdict(list)
        with open(tracks_csv, "r") as f:
            for r in csv.DictReader(f):
                boxes_by_frame[int(r["frame"])].append(
                    (int(r["id"]), (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))))

        shuttle_coords = defaultdict(lambda: [0, 0])
        with open(shuttle_csv, "r") as f:
            for r in csv.DictReader(f):
                shuttle_coords[int(r["Frame"])] = [int(r["X"]), int(r["Y"])]

        events, contacts = load_events_and_contacts(events_json)
        event_by_frame = defaultdict(dict)
        for e in events:
            for fr in range(int(e["t0"]), int(e["t1"]) + 1):
                event_by_frame[fr][int(e["track_id"])] = e

        # Video Prep
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FFmpeg Pipeline
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{W}x{H}', r=fps)
            .output(out_path, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        # Configs for drawing
        scale = max(H, W) / 1080.0
        BOX_TH = max(3, int(6 * scale))
        FONT_SCALE = max(0.8, 0.9 * scale)
        PAD = max(6, int(8 * scale))

        for i in tqdm(range(total_frames), desc="Overlay"):
            ok, frame = cap.read()
            if not ok: break
            
            overlay = frame.copy()

            # Shuttle
            sc = shuttle_coords[i]
            if sc != [0,0]:
                cv2.circle(overlay, tuple(sc), 10, _color_for_id(5), 5)

            # Frame No
            cv2.putText(overlay, f"Frame {i}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 200), 4)

            # Contact
            if i in contacts or i-1 in contacts or i+1 in contacts:
                cv2.putText(overlay, "Contact", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE*1.2, (0, 0, 255), 4)

            # Players
            for tid, (x1, y1, x2, y2) in boxes_by_frame.get(i, []):
                color = _color_for_id(tid)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, BOX_TH)
                
                ev = event_by_frame.get(i, {}).get(tid)
                if ev and ev['label'] != "negative":
                    text = f"{ev['label']} {ev['p']:.2f}"
                    
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
                    tx, ty = x1, y1 - PAD
                    if ty - th - PAD < 0: ty = y1 + th + PAD
                    
                    cv2.rectangle(overlay, (tx, ty - th - PAD), (tx + tw, ty), color, -1)
                    cv2.putText(overlay, text, (tx, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,255), 2)

            process.stdin.write(overlay.tobytes())

        cap.release()
        process.stdin.close()
        process.wait()

        return ProcessingResult(True, {'overlay_video': out_path}, processing_time=time.time() - start_time)

    except Exception as e:
        return ProcessingResult(False, None, str(e))