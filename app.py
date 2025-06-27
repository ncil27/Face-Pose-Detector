from flask import Flask, render_template, request, jsonify,  redirect, url_for
import os, cv2, joblib, numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename
from PIL import Image
import base64
# from flask import Flask, render_template, request
# from PIL import Image
from io import BytesIO
# from flask import Flask, render_template,
import subprocess
# import numpy as np
# import cv2

import base64
from io import BytesIO

app = Flask(__name__)

# Load model
model = joblib.load("models/pose_model.joblib")
pca = joblib.load("models/pca_transformer.joblib")
le = joblib.load("models/label_encoder.joblib")

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

IMG_WIDTH = 64
IMG_HEIGHT = 60

@app.route('/')
def index():
    print("Index page accessed")
    return render_template('index.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/webcam_capture')
def webcam_capture():
    return render_template('webcam_capture.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    in_memory_file = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if not results.detections:
        return jsonify({'error': 'No face detected'})

    bboxC = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    bw = int(bboxC.width * w)
    bh = int(bboxC.height * h)

    x = max(0, x)
    y = max(0, y)
    bw = min(w - x, bw)
    bh = min(h - y, bh)

    face = img[y:y+bh, x:x+bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    flat = resized.flatten().reshape(1, -1) / 255.0
    transformed = pca.transform(flat)
    prediction = model.predict(transformed)
    label = le.inverse_transform(prediction)[0]

    return jsonify({'label': label})


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.get_json()
    img_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes)).convert('L')  # Convert ke grayscale

    img = img.resize((64, 60))  # Ukuran sesuai training
    img_np = np.array(img).flatten().reshape(1, -1) / 255.0
    img_pca = pca.transform(img_np)
    prediction = model.predict(img_pca)
    label = le.inverse_transform(prediction)[0]

    return jsonify({'label': label})

@app.route('/open_camera')
def open_camera():
    # Jalankan script Python yang real-time
    subprocess.Popen(["python", "pose_predictor.py"])
    return redirect(url_for('menu'))  # Kembali ke menu setelah dijalankan


if __name__ == '__main__':
    app.run(debug=True)
