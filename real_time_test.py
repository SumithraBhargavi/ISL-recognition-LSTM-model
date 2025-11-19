from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ---------- Config ----------
MODEL_PATH = "dataset/isl_lstm_model.keras"  # path to your trained model
SEQ_LEN = 30
CONFIDENCE_THRESHOLD = 0.7
labels = ['Help', 'Hello', 'Thank_you']  # same as training

# ---------- Load model ----------
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------- Setup MediaPipe ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Initialize variables ----------
sequence = deque(maxlen=SEQ_LEN)  # stores last 30 frames
predicted_gesture = "None"

# ---------- Start webcam ----------
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ---------- Extract landmarks ----------
        if results.multi_hand_landmarks:
            row = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row += [0.0] * (126 - len(row))  # pad missing features
            sequence.append(row)

            # ---------- Make prediction ----------
            if len(sequence) == SEQ_LEN:
                X = np.expand_dims(np.array(sequence), axis=0)  # shape: (1,30,126)
                y_pred = model.predict(X, verbose=0)
                max_prob = np.max(y_pred)
                predicted_gesture = labels[np.argmax(y_pred)] if max_prob > CONFIDENCE_THRESHOLD else "None"

        else:
            # No hand detected: clear sequence
            sequence.clear()
            predicted_gesture = "None"

        # ---------- Display ----------
        cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ISL Gesture Recognition", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
