import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os

# Emotion dictionary
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load model structure
model_path = "model/emotion_model.json"
weights_path = "model/emotion_model.h5"

if not os.path.exists(model_path) or not os.path.exists(weights_path):
    raise FileNotFoundError("Model files not found. Ensure 'emotion_model.json' and 'emotion_model.h5' exist in the 'model' directory.")

with open(model_path, 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)

# Load model weights
emotion_model.load_weights(weights_path)
print("✅ Loaded model from disk")

# Start the video feed
video_path = "emotion_sample6.mp4"  # Change to 0 for webcam

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"❌ Error: Could not open video file {video_path}")

# Load Haar cascade for face detection
haar_cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(haar_cascade_path):
    raise FileNotFoundError("Haar cascade XML file not found. Ensure 'haarcascade_frontalface_default.xml' is in the correct directory.")

face_detector = cv2.CascadeClassifier(haar_cascade_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video ended or cannot be read.")
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract face region and preprocess
        roi_gray = gray_frame[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)  # Reshape for model

        try:
            # Predict emotion
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[maxindex]

            # Display detected emotion
            cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"⚠️ Error in prediction: {e}")

    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
