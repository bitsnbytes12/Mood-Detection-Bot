from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
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

# Initialize MTCNN
detector = MTCNN()

# Load Haar Cascade for smile detection
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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

last_spoken_times = {emotion: 0 for emotion in emotion_labels}  # Per mood delay
speech_delay = 4  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        faces = detector.detect_faces(rgb_frame)
    except Exception as e:
        print(f"Face detection error: {e}")
        faces = []

    # Process only the first face to avoid overlapping texts
    if faces:
        face = faces[0]
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        face_roi = rgb_frame[y:y+h, x:x+w]
        if face_roi.size != 0:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray / 255.0
            gray = np.reshape(gray, (1, 48, 48, 1))

            preds = emotion_model.predict(gray, verbose=0)
            max_prob = np.max(preds)
            emotion = emotion_labels[np.argmax(preds)]
            # Additional rules for distinct moods based on facial features
            keypoints = face['keypoints']
            nose_y = keypoints['nose'][1]
            mouth_left_y = keypoints['mouth_left'][1]
            mouth_right_y = keypoints['mouth_right'][1]
            mouth_center_y = (mouth_left_y + mouth_right_y) / 2
            mouth_open = abs(mouth_left_y - mouth_right_y) > 5

            # Refine Neutral to Sad if mouth is downturned
            if emotion == 'Neutral' and mouth_center_y > nose_y + 10:
                emotion = 'Sad'

            # Refine Neutral to Surprise if mouth is open
            if emotion == 'Neutral' and mouth_open:
                emotion = 'Surprise'

            # Refine Neutral to Angry if mouth is closed and no smile
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            if emotion == 'Neutral' and not mouth_open and len(smiles) == 0:
                emotion = 'Angry'

            # Refine Neutral to Disgust if upper lip is raised
            if emotion == 'Neutral' and mouth_left_y < nose_y - 5:
                emotion = 'Disgust'
            # Apply overrides only if model confidence is low
            if max_prob < 0.5:
                # Check for smile to override to Happy
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
                if len(smiles) > 0:
                    emotion = 'Happy'

                # Check for widened eyes to override to Fear
                keypoints = face['keypoints']
                left_eye_y = keypoints['left_eye'][1]
                right_eye_y = keypoints['right_eye'][1]
                avg_eye_y = (left_eye_y + right_eye_y) / 2
                if avg_eye_y > y + h * 0.4:  # If eyes are positioned low in the face (approximating widened)
                    emotion = 'Fear'

            # NLP + API Response
            response = get_transformer_response(emotion)

            # ===============================
            # Speak the response (with delay per mood)
            # ===============================
            current_time = time.time()
            if current_time - last_spoken_times[emotion] > speech_delay:
                speak(response)
                last_spoken_times[emotion] = current_time

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
