##############################################################################
# API endpoints.
##############################################################################

import os
import tempfile
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
from app.core.pipeline import process_video_pipeline
from app.core.model_loader import model_manager
from app.utils.gcs import download_blob, upload_blob
from app.config import settings

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(model_manager.device),
        "models_loaded": {
            "yolo": model_manager.yolo_model is not None,
            "strongsort": model_manager.strongsort_tracker is not None,
            "contact": model_manager.contact_model is not None,
            "slowfast": model_manager.slowfast_model is not None
        }
    })

@api_bp.route('/process', methods=['POST'])
def process():
    video_file = request.files.get('video')
    gcs_uri = request.form.get('gcs_uri')
    output_bucket = request.form.get('output_bucket')
    user_id = request.form.get('user_id', 'unknown')
    job_id = request.form.get('job_id', 'unknown')
    
    if not video_file and not gcs_uri:
        return jsonify({"error": "No input provided"}), 400

    # Temp File Management
    # delete=False because we need to close it before different threads/subprocesses access it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        input_path = tmp.name

    try:
        if video_file:
            video_file.save(input_path)
        else:
            download_blob(gcs_uri, input_path)

        output_dir = tempfile.mkdtemp(dir=settings.OUTPUTS_DIR)
        print(f"🚀 Processing: {input_path} -> {output_dir}")
        
        result = process_video_pipeline(input_path, output_dir, request.form)

        if not result['success']:
            return jsonify(result), 500

        # Upload Outputs
        uploaded_files = {}
        if output_bucket:
            base_blob = f"outputs/{user_id}/{job_id}"
            
            to_upload = {
                'tracks_csv': result.get('tracks_csv'),
                'shuttle_csv': result.get('shuttle_csv'),
                'overlay_video': result.get('overlay_video')
            }
            
            for key, local_path in to_upload.items():
                if local_path and os.path.exists(local_path):
                    fname = os.path.basename(local_path)
                    uri = upload_blob(Path(local_path), output_bucket, f"{base_blob}/{fname}")
                    if uri: uploaded_files[key] = uri

        return jsonify({
            "job": {"user_id": user_id, "job_id": job_id},
            "events": result.get('events', []),
            "outputs": uploaded_files,
            "timing": result.get('timing')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

@api_bp.route('/download', methods=['GET'])
def download():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)