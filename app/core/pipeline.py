###############################################################################
# Main orchestrator for all steps (including concurrent).
###############################################################################

import threading
import time
from queue import Queue
from app.schemas import YoloCfg, InferenceCfg, SlowFastInferCfg, TrainCfg
from app.services.tracking import run_strongsort
from app.services.shuttle import run_tracknet
from app.services.contact import run_contact_detection
from app.services.action import run_action_recognition
from app.services.visualization import run_overlay
from app.core.model_loader import model_manager
from app.config import settings

def process_video_pipeline(video_path: str, output_dir: str, params: dict):
    start_time = time.time()
    result_queue = Queue()
    
    yolo_cfg = YoloCfg() # Default config

    print("🔄 Pipeline Steps 1 & 2 (Concurrent)...")
    
    t1 = threading.Thread(
        target=lambda q: q.put(('strongsort', run_strongsort(video_path, output_dir, yolo_cfg, model_manager))),
        args=(result_queue,)
    )
    t2 = threading.Thread(
        target=lambda q: q.put(('tracknet', run_tracknet(
            video_path, output_dir, 
            int(params.get('batch_size', 16)), 
            params.get('eval_mode', 'weight')
        ))),
        args=(result_queue,)
    )

    parallel_start = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    parallel_time = time.time() - parallel_start

    results = {}
    while not result_queue.empty():
        k, v = result_queue.get()
        results[k] = v

    if not results['strongsort'].success: return {'success': False, 'error': f"StrongSort failed: {results['strongsort'].error}"}
    if not results['tracknet'].success: return {'success': False, 'error': f"TrackNet failed: {results['tracknet'].error}"}

    tracks_csv = results['strongsort'].data['tracks_csv']
    shuttle_csv = results['tracknet'].data['shuttle_csv']

    # Step 3: Contact
    print("💥 Step 3: Contact Detection")
    contact_res = run_contact_detection(shuttle_csv, model_manager.contact_model, model_manager.device)
    contact_frames = contact_res.data['contact_frames'] if contact_res.success else []

    # Step 4: Action
    print("🎬 Step 4: Action Recognition")
    events = []
    events_json = ""
    inference_cfg = InferenceCfg(settings.LABELS, SlowFastInferCfg())
    
    if contact_res.success:
        action_res = run_action_recognition(video_path, tracks_csv, contact_frames, output_dir, model_manager.slowfast_model, inference_cfg)
        if action_res.success:
            events = action_res.data['events']
            events_json = action_res.data['events_json']

    # Step 5: Overlay
    print("🎨 Step 5: Overlay")
    overlay_res = run_overlay(video_path, tracks_csv, shuttle_csv, events_json, contact_frames, output_dir)
    
    total_time = time.time() - start_time
    time_saved = (results['strongsort'].processing_time + results['tracknet'].processing_time) - parallel_time

    return {
        'success': True,
        'tracks_csv': tracks_csv,
        'shuttle_csv': shuttle_csv,
        'overlay_video': overlay_res.data.get('overlay_video') if overlay_res.success else None,
        'events': events,
        'timing': {
            'total': total_time,
            'saved': time_saved,
            'parallel_stage': parallel_time
        }
    }