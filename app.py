import cv2
import joblib
import numpy as np
import mediapipe as mp # Import Mediapipe
import sys # Tambahkan ini untuk sys.exit()

# --- PENGATURAN DAN PEMUATAN MODEL ---
IMG_HEIGHT = 60
IMG_WIDTH = 64

# 1. Muat semua aset yang sudah disimpan
try:
    model = joblib.load('models/pose_model.joblib')
    pca = joblib.load('models/pca_transformer.joblib')
    le = joblib.load('models/label_encoder.joblib')
    print("Model dan semua aset berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file model.joblib, pca_transformer.joblib, dan label_encoder.joblib ada di folder /models.")
    sys.exit() # Gunakan sys.exit() untuk keluar dari script

# --- Inisialisasi Mediapipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Parameter deteksi wajah. min_detection_confidence bisa disesuaikan (0.5-1.0)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) # model_selection=1 untuk jarak jauh
print("Mediapipe Face Detector berhasil diinisialisasi.")
# -----------------------------------------------

# 3. Inisialisasi webcam
cap = cv2.VideoCapture(0) # Angka 0 biasanya untuk webcam internal

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    sys.exit()

print("\nKamera siap. Arahkan wajah ke kamera.")
print("Tekan 'q' pada jendela video untuk keluar.")

# --- LOOP UTAMA APLIKASI ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera. Keluar...")
        break

    # Penting: Mediapipe menggunakan format RGB, jadi konversi BGR ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah menggunakan Mediapipe
    results = face_detection.process(rgb_frame)

    # Jika ada wajah terdeteksi
    if results.detections:
        for detection in results.detections:
            # Ekstrak bounding box dari deteksi Mediapipe
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # --- Pastikan bounding box valid dan tidak keluar dari frame ---
            x = max(0, x)
            y = max(0, y)
            w = min(iw - x, w)
            h = min(ih - y, h)
            # Tambahkan margin ke bounding box
            margin_x = int(w * 0.15) # Tambah 15% lebar sebagai margin horizontal
            margin_y = int(h * 0.15) # Tambah 15% tinggi sebagai margin vertikal

            x_new = max(0, x - margin_x)
            y_new = max(0, y - margin_y)
            w_new = min(iw - x_new, w + 2 * margin_x) # Lebar asli + margin kiri + margin kanan
            h_new = min(ih - y_new, h + 2 * margin_y) # Tinggi asli + margin atas + margin bawah

            # Perbarui variabel x, y, w, h
            x, y, w, h = x_new, y_new, w_new, h_new

            if w <= 0 or h <= 0:
                continue
            # -------------------------------------------------------------

            # Gambar kotak di sekeliling wajah yang terdeteksi pada frame asli
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop wajah dari frame (gunakan frame asli, karena model Anda dilatih pada grayscale)
            # Namun, pastikan cropping dari grayscale_frame agar konsisten
            gray_frame_for_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray_frame_for_crop[y:y+h, x:x+w]

            # --- PREPROCESSING WAJAH (HARUS SAMA PERSIS DENGAN SAAT TRAINING) ---
            # Pastikan face_roi tidak kosong setelah cropping
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                cv2.putText(frame, "ROI Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                continue # Lewati jika ROI tidak valid

            img_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
            img_flattened = img_resized.flatten()
            img_scaled = np.array(img_flattened).reshape(1, -1) / 255.0
            img_pca = pca.transform(img_scaled)

            # --- PREDIKSI POSE ---
            prediction_numeric = model.predict(img_pca)
            prediction_label = le.inverse_transform(prediction_numeric)[0]
            
            # Tampilkan teks hasil prediksi di atas kotak wajah
            cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame hasil akhir ke jendela
    cv2.imshow('Deteksi Pose Wajah (Mediapipe)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- BERSIH-BERSIH ---
cap.release()
cv2.destroyAllWindows()
face_detection.close() # Tutup objek Mediapipe
print("Aplikasi ditutup.")