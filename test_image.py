import cv2
import joblib
import numpy as np
import sys
import os
import mediapipe as mp # Import Mediapipe

# --- PENGATURAN DAN PEMUATAN MODEL ---
IMG_HEIGHT = 60
IMG_WIDTH = 64

# Muat semua aset yang sudah disimpan
try:
    model = joblib.load('models/pose_model.joblib')
    pca = joblib.load('models/pca_transformer.joblib')
    le = joblib.load('models/label_encoder.joblib')
    print("Model dan semua aset berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file model.joblib, pca_transformer.joblib, dan label_encoder.joblib ada di folder /models.")
    sys.exit()

# --- Inisialisasi Mediapipe Face Detection (dilakukan hanya sekali) ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
print("Mediapipe Face Detector berhasil diinisialisasi.")
# -----------------------------------------------

# --- FUNGSI DETEKSI DAN PREDIKSI ---
def predict_pose_from_image(image_path):
    print(f"\n--- Menguji {image_path} dengan Mediapipe ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak bisa membaca gambar dari {image_path}. Pastikan path benar.")
        return

    display_img = img.copy()
    
    # Penting: Mediapipe menggunakan format RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi wajah menggunakan Mediapipe
    results = face_detection.process(rgb_img)

    if not results.detections:
        print("Tidak ada wajah terdeteksi dalam gambar ini.")
        cv2.imshow(f"Original Image (No Face Detected) - {image_path}", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)

        # --- Pastikan bounding box valid dan tidak keluar dari frame ---
        x = max(0, x)
        y = max(0, y)
        w = min(iw - x, w)
        h = min(ih - y, h)
        
        if w <= 0 or h <= 0:
            print(f"Peringatan: ROI wajah tidak valid ({w}x{h}), mungkin deteksi salah.")
            continue

        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop wajah dari gambar grayscale untuk input model
        gray_img_for_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = gray_img_for_crop[y:y+h, x:x+w]

        # --- PREPROCESSING WAJAH ---
        if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
            cv2.putText(display_img, "ROI Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print("Peringatan: ROI wajah kosong setelah cropping.")
            continue

        img_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
        img_flattened = img_resized.flatten()
        img_scaled = np.array(img_flattened).reshape(1, -1) / 255.0
        img_pca = pca.transform(img_scaled)

        # --- PREDIKSI POSE ---
        prediction_numeric = model.predict(img_pca)
        prediction_label = le.inverse_transform(prediction_numeric)[0]
        
        cv2.putText(display_img, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Pose terdeteksi: {prediction_label}")

    cv2.imshow(f'Deteksi Pose Wajah dari Gambar - {image_path}', display_img)
    print("Tekan sembarang tombol untuk keluar dari jendela gambar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # face_detection.close() # <-- BARIS INI DIHAPUS DARI SINI

# --- PENGGUNAAN ---
if __name__ == "__main__":
    photo_straight = 'photo_straight.png'
    photo_left = 'photo_left.png'
    photo_right = 'photo_right.png'
    photo_up = 'photo_up.png' # Ini adalah foto dengan wajah mendongak ke atas

    # Kita akan coba lagi photo_up (mendongak ke atas) dengan confidence yang lebih rendah
    print("\n--- Menguji photo_up.png (wajah mendongak ke atas) dengan Mediapipe (Confidence 0.5) ---")
    temp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    original_face_detection = face_detection
    face_detection = temp_face_detection
    
    predict_pose_from_image(photo_up)
    
    face_detection = original_face_detection
    temp_face_detection.close()

    # Uji foto lainnya seperti biasa
    predict_pose_from_image(photo_straight)
    predict_pose_from_image(photo_left)
    predict_pose_from_image(photo_right)

    face_detection.close()
    print("\nSelesai menguji.")