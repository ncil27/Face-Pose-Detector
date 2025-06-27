from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import mediapipe as mp
import base64

app = Flask(__name__)

# Load model, PCA, dan LabelEncoder
model = joblib.load('models/pose_model.joblib')
pca = joblib.load('models/pca_transformer.joblib')
le = joblib.load('models/label_encoder.joblib')

IMG_HEIGHT, IMG_WIDTH = 60, 64
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

def process_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            # Tambahkan margin
            margin_x, margin_y = int(w * 0.15), int(h * 0.15)
            x, y = max(0, x - margin_x), max(0, y - margin_y)
            w = min(iw - x, w + 2 * margin_x)
            h = min(ih - y, h + 2 * margin_y)

            if w <= 0 or h <= 0:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            img_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
            img_flat = img_resized.flatten().reshape(1, -1) / 255.0
            img_pca = pca.transform(img_flat)
            pred = model.predict(img_pca)
            label = le.inverse_transform(pred)[0]

            return label

    return "No face detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    label = process_image(image)
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
