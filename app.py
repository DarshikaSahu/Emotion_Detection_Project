from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load Pretrained Model
model = load_model('model/emotion_model.h5')

# Emotion Labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion Detection Function
def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return emotion_labels[np.argmax(prediction)]

# API Endpoint to Detect Emotion from Image
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    img_path = "temp.jpg"
    img_file.save(img_path)

    detected_emotion = predict_emotion(img_path)
    return jsonify({"emotion": detected_emotion})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
