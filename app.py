import traceback

from flask import Flask, render_template, request, jsonify, url_for
import base64
import re
import os
from model import load_model_binary, load_model_multi, predict_xray_binary, predict_xray_multi, ensure_model_directory
from werkzeug.utils import secure_filename
from PIL import Image
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ensure_model_directory()

model_binary = load_model_binary()
model_multi = load_model_multi()
if model_binary is None or model_multi is None:
    print("WARNING: Running with mock model.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'filepath': filepath
        })

    return jsonify({'error': 'Invalid file type. Please upload an image (JPG, PNG)'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath:
        return jsonify({'error': 'No filepath provided'}), 400

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    with open(filepath, 'rb') as f:
        img_bytes = f.read()

    if img_bytes is None:
        return jsonify({'error': 'Failed to read image'}), 500

    results = predict_xray_binary(model_binary, img_bytes)

    return jsonify({
        'success': True,
        'hasDisease': results
    })

@app.route('/detailed-analysis', methods=['POST'])
def detailed_analysis():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath:
        return jsonify({'error': 'No filepath provided'}), 400

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    with open(filepath, 'rb') as f:
        img_bytes = f.read()

    if img_bytes is None:
        return jsonify({'error': 'Failed to read image'}), 500

    results = predict_xray_multi(model_multi, img_bytes)

    return jsonify({
        'success': True,
        'diseases': results
    })

@app.route('/model-status')
def model_status():
    """Endpoint to check if the model is loaded"""
    return jsonify({
        "model_loaded": model_binary is not None,
        "model_type": "CheXNet" if model_binary is not None else "Mock Model"
    })