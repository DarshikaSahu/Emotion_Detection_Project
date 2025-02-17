import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import nltk
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from nltk.sentiment import SentimentIntensityAnalyzer

# Load model and NLP
model = load_model('model/emotion_model.h5')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
engine = pyttsx3.init()

# Emotion categories
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion-based music recommendations
music_dict = {
    'happy': 'play_happy.mp3',
    'sad': 'play_sad.mp3',
    'angry': 'play_angry.mp3',
    'neutral': 'play_neutral.mp3',
    'surprise': 'play_surprise.mp3'
}

# It'll be a sad day when you leave us
# I'd do anything to make her happy

# Speech Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("🗣️ You said:", text)
        return text
    except:
        print("❌ Could not recognize speech")
        return None

# Text Sentiment Analysis
def analyze_text_sentiment(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return 'happy'
    elif sentiment['compound'] <= -0.05:
        return 'sad'
    else:
        return 'neutral'

# Image Emotion Detection
# img_path = "https://blog.stocksnap.io/content/images/2022/02/smiling-woman_W6GFOSFAXA.jpg2"
# Today is a beautiful day to embrace happiness and spread smiles!

def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return emotion_labels[np.argmax(prediction)]

# Real-Time Emotion Detection from Webcam
def live_emotion_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128)) / 255.0
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face)
            detected_emotion = emotion_labels[np.argmax(prediction)]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Real-Time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Detect emotion from live webcam")
    print("2. Detect emotion from an image")
    print("3. Analyze text sentiment")
    print("4. Detect emotion from speech")
    choice = input("Enter option: ")

    if choice == '1':
        live_emotion_detection()
    elif choice == '2':
        img_path = input("Enter image path: ")
        emotion = predict_emotion(img_path)
        print(f"Detected Emotion: {emotion}")
    elif choice == '3':
        text = input("Enter text: ")
        emotion = analyze_text_sentiment(text)
        print(f"Detected Emotion: {emotion}")
    elif choice == '4':
        text = recognize_speech()
        if text:
            emotion = analyze_text_sentiment(text)
            print(f"Detected Emotion: {emotion}")
    else:
        print("Invalid choice!")

        # I wish someone would notice the sadness behind my smile
