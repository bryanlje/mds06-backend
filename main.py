import os
from flask import Flask
from app.api.routes import api_bp
from app.core.model_loader import model_manager

def create_app():
    # Trigger model loading on startup
    print("Initializing Application...")
    _ = model_manager 
    
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, threaded=True)