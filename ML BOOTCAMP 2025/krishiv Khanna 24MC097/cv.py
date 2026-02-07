import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the model
model_path = r"C:\Users\krish\OneDrive\Desktop\fastapi\myenv\my_model.keras"  # or .keras
model = load_model(model_path)

# Class index to label mapping (must match train_generator.class_indices)
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = preprocess_input(face.astype('float32'))
        face = np.expand_dims(face, axis=0)

        # Predict
        preds = model.predict(face)[0]
        pred_label = class_labels[np.argmax(preds)]
        confidence = np.max(preds)

        # Draw
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{pred_label} ({confidence:.2f})"
        cv2.putText(orig_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Optional: Show probabilities
        print("Predictions:", {label: round(prob, 2) for label, prob in zip(class_labels, preds)})

    cv2.imshow('Emotion Detection', orig_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
