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

# @app.route('/analyze', methods=['POST'])
# def analyze_xray():
#     """
#     Process uploaded X-ray images and return analysis results
#     """
#     saved_file_path = None
#     print(request.files['xray_image'])
#     print(request.form)
#     try:
#         if 'xray_image' in request.files:
#             # Handle file upload
#             file = request.files['xray_image']
#             if file.filename == '':
#                 return jsonify({"error": "No file selected"}), 400
#
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 saved_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(saved_file_path)
#
#                 # Process with model
#                 with open(saved_file_path, 'rb') as f:
#                     img_bytes = f.read()
#
#                 results = predict_xray(model, img_bytes)
#                 print(results)
#                 # Return results with the saved image path
#                 return jsonify({
#                     "results": results,
#                     "image_url": url_for('static', filename=f'uploads/{filename}')
#                 })
#             else:
#                 return jsonify({"error": "Invalid file format. Please upload JPG, JPEG, or PNG"}), 400
#
#         elif 'image_data' in request.form:
#             # Handle base64 encoded image
#             image_data = request.form['image_data']
#
#             # Remove data URL prefix if present
#             if 'base64,' in image_data:
#                 image_data = re.sub('^data:image/.+;base64,', '', image_data)
#
#             img_bytes = base64.b64decode(image_data)
#
#             # Save the image
#             filename = f"capture_{request.remote_addr.replace('.', '_')}_{int(time.time())}.jpg"
#             saved_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#
#             with open(saved_file_path, 'wb') as f:
#                 f.write(img_bytes)
#
#             # Process with model
#             results = predict_xray(model, img_bytes)
#
#             # Return results with the saved image path
#             return jsonify({
#                 "results": results,
#                 "image_url": url_for('static', filename=f'uploads/{filename}')
#             })
#
#         else:
#             return jsonify({"error": "No image provided"}), 400
#
#     except Exception as e:
#         # Log the error
#         print(f"Error processing image: {str(e)}")
#         traceback.print_exc()
#         # Clean up saved file if there was an error
#         if saved_file_path and os.path.exists(saved_file_path):
#             try:
#                 os.remove(saved_file_path)
#             except:
#                 pass
#
#         return jsonify({"error": f"Error processing image: {str(e)}"}), 500

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

if __name__ == '__main__':
    import time
    app.run(debug=True)