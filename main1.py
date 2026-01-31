from tensorflow.keras.models import load_model
from nlp_engine import get_transformer_response
import pyttsx3
import time
import cv2
import numpy as np
import threading

# ===============================

# ===============================
# Initialize TTS Engine
# ===============================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 165)     # Speech speed
tts_engine.setProperty('volume', 1.0)   # Volume (0.0 to 1.0)

# ===============================
# Load Emotion Model
# ===============================
emotion_model = load_model("emotion_model.h5")
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Conventions for each mood
mood_conventions = {
    'Angry': 'Furrowed brows, narrowed eyes, clenched jaw.',
    'Disgust': 'Wrinkled nose, raised upper lip, narrowed eyes.',
    'Fear': 'Wide eyes, raised brows, open mouth.',
    'Happy': 'Raised cheeks, smiling eyes, upturned mouth.',
    'Sad': 'Downward brows, drooping eyes, downturned mouth.',
    'Surprise': 'Raised brows, wide eyes, open mouth.',
    'Neutral': 'Relaxed features, no strong expression.'
}

# Webcam
cap = cv2.VideoCapture(0)
print("Press Q to exit")

# Set up floating window
cv2.namedWindow("Face Mood NLP Assistant (Voice Enabled)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Mood NLP Assistant (Voice Enabled)", 640, 480)
cv2.moveWindow("Face Mood NLP Assistant (Voice Enabled)", 100, 100)
cv2.setWindowProperty("Face Mood NLP Assistant (Voice Enabled)", cv2.WND_PROP_TOPMOST, 1)

def speak(response):
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    engine.say(response)
    engine.runAndWait()

last_spoken_time = 0
speech_delay = 3  # seconds (prevents repeated speaking)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process only the first face to avoid overlapping texts
    if len(faces) > 0:
        x, y, w, h = faces[0]

        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size != 0:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray / 255.0
            gray = np.reshape(gray, (1, 48, 48, 1))

            preds = emotion_model.predict(gray, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]

            # NLP + API Response
            response = get_transformer_response(emotion)

            # ===============================
            # Speak the response (with delay)
            # ===============================
            current_time = time.time()
            if current_time - last_spoken_time > speech_delay:
                tts_engine.say(response)
                tts_engine.runAndWait()
                last_spoken_time = current_time

            # UI Drawing
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Mood: {emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Display convention for the detected mood
            convention = mood_conventions.get(emotion, 'Unknown mood.')
            cv2.putText(frame, f"Convention: {convention}",
                        (10, frame.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            cv2.putText(frame, "AI-powered emotion detection for better interaction.",
                        (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            # UI Drawing
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Mood: {emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Display convention for the detected mood
            convention = mood_conventions.get(emotion, 'Unknown mood.')
            cv2.putText(frame, f"Convention: {convention}",
                        (10, frame.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            cv2.putText(frame, "AI-powered emotion detection for better interaction.",
                        (10, frame.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    cv2.imshow("Face Mood NLP Assistant (Voice Enabled)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
