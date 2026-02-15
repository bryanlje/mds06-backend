###############################################################################
# Subprocess wrapper for TrackNet.
###############################################################################

import sys
import subprocess
import os
import time
import csv
from app.config import settings
from app.schemas import ProcessingResult

def read_tracknet_csv(csv_path: str):
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_dict['Frame'].append(int(row['Frame']))
            pred_dict['X'].append(int(row['X']))
            pred_dict['Y'].append(int(row['Y']))
            pred_dict['Visibility'].append(int(row['Visibility']))
    return pred_dict

def run_tracknet(video_path: str, output_dir: str, batch_size: int, eval_mode: str):
    start_time = time.time()
    try:
        video_name = os.path.basename(video_path).split(".")[0]
        
        # Ensure TrackNet module is in Python Path
        if str(settings.TRACKNET_REPO) not in sys.path:
            sys.path.insert(0, str(settings.TRACKNET_REPO))

        cmd = [
            sys.executable, "-m", "TrackNetV3.predict",
            "--video_file", video_path,
            "--tracknet_file", settings.TRACKNET_WEIGHTS,
            "--inpaintnet_file", settings.INPAINTNET_WEIGHTS,
            "--save_dir", output_dir,
            "--batch_size", str(batch_size),
            "--eval_mode", eval_mode
        ]

        print(f"[TrackNet] Running subprocess: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, universal_newlines=True
        )
        
        # Consume stdout to prevent buffer blocking
        for line in process.stdout:
            print(f"[TrackNet] {line.strip()}")
        
        return_code = process.wait()
        if return_code != 0:
            error_msg = process.stderr.read()
            raise RuntimeError(f"TrackNet failed: {error_msg}")
        
        expected_csv = os.path.join(output_dir, f"{video_name}_ball.csv")
        if not os.path.exists(expected_csv):
            raise FileNotFoundError(f"TrackNet did not generate expected CSV: {expected_csv}")

        # Optional: Read results back for verification or API return
        pred_dict = read_tracknet_csv(expected_csv)

        return ProcessingResult(
            True, 
            {'shuttle_csv': expected_csv, 'predictions': pred_dict}, 
            processing_time=time.time() - start_time
        )
    except Exception as e:
        print(f"[TrackNet] Error: {e}")
        return ProcessingResult(False, None, str(e))