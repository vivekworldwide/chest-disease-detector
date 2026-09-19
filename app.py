import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from model import ChestDiseaseClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Initialize the trained PyTorch inference classifier
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest_disease_model.pth")
classifier = None

try:
    print(f"Loading Chest Disease AI Model from: {MODEL_PATH}")
    classifier = ChestDiseaseClassifier(MODEL_PATH)
    print(f"Model successfully loaded on device: {classifier.device}")
except Exception as e:
    print(f"Warning: Model could not be loaded immediately ({e}). It will retry on request.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier is not None,
        "classes": classifier.classes if classifier else [],
        "device": str(classifier.device) if classifier else "unknown"
    })


@app.route('/sample/<path:filename>')
def serve_sample(filename):
    samples_dir = os.path.join(app.root_path, 'static', 'samples')
    return send_from_directory(samples_dir, filename)


@app.route('/predict', methods=['POST'])
def predict():
    global classifier
    if classifier is None:
        try:
            classifier = ChestDiseaseClassifier(MODEL_PATH)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Model failed to initialize: {str(e)}"
            }), 500

    image_bytes = None
    source_name = "Uploaded Scan"

    # Case 1: File uploaded via multipart/form-data
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No image file selected."}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "success": False, 
                "error": "Unsupported file format. Please upload a PNG, JPG, or JPEG chest X-ray image."
            }), 400
        
        source_name = secure_filename(file.filename)
        image_bytes = file.read()

    # Case 2: JSON payload with sample name or base64
    elif request.is_json:
        data = request.get_json()
        if 'sample' in data:
            sample_file = data['sample']
            sample_path = os.path.join(app.root_path, 'static', 'samples', secure_filename(sample_file))
            if not os.path.exists(sample_path):
                return jsonify({"success": False, "error": f"Sample image {sample_file} not found."}), 404
            
            source_name = sample_file
            with open(sample_path, 'rb') as f:
                image_bytes = f.read()

        elif 'image_base64' in data:
            try:
                base64_str = data['image_base64']
                if ',' in base64_str:
                    base64_str = base64_str.split(',', 1)[1]
                image_bytes = base64.b64decode(base64_str)
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64 image data: {str(e)}"}), 400

    if not image_bytes:
        return jsonify({"success": False, "error": "No valid image data received."}), 400

    try:
        # Validate that image is readable and convert to base64 preview for UI
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare small base64 thumbnail for UI confirmation
        preview_buffer = io.BytesIO()
        pil_img.save(preview_buffer, format="JPEG", quality=85)
        preview_base64 = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode('utf-8')

        # Run AI prediction
        result = classifier.predict(pil_img)

        return jsonify({
            "success": True,
            "filename": source_name,
            "image_preview": preview_base64,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
            "device": result["device"],
            "probabilities": result["probabilities"],
            "metadata": result["metadata"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error during chest X-ray processing: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
