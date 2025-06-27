import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('path_to_your_saved_model.h5')

# Define pose classes based on your training
POSE_CLASSES = {
    0: "Front",
    1: "Left",
    2: "Right",
    # Add more as per your model
}

def predict_pose(image):
    # Preprocess the image as you did in training
    resized = cv2.resize(image, (224, 224))  # Adjust size as needed
    normalized = resized / 255.0
    input_array = np.expand_dims(normalized, axis=0)
    
    # Make prediction
    predictions = model.predict(input_array)
    predicted_class = np.argmax(predictions)
    
    return POSE_CLASSES[predicted_class]