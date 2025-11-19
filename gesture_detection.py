import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# ===============================
# CONFIG
# ===============================
MODEL_PATH = os.path.join("dataset", "isl_lstm_model.keras")
LABELS_PATH = os.path.join("dataset", "labels.npy")
SEQ_LEN = 50

# ===============================
# LOAD MODEL AND LABELS
# ===============================
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)
print("✅ Model and labels loaded successfully!")

# ===============================
# MEDIAPIPE SETUP
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===============================
# START WEBCAM
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

seq = []

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            row = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pad or truncate to 126
            if len(row) < 126:
                row += [0.0] * (126 - len(row))
            elif len(row) > 126:
                row = row[:126]

            seq.append(row)
            if len(seq) > SEQ_LEN:
                seq.pop(0)

            # Only predict once we have enough frames
            if len(seq) == SEQ_LEN:
                input_seq = np.expand_dims(seq, axis=0)
                preds = model.predict(input_seq, verbose=0)
                pred_idx = np.argmax(preds)
                gesture = labels[pred_idx]
                confidence = preds[0][pred_idx]

                # Show prediction
                cv2.putText(frame, f'{gesture} ({confidence*100:.1f}%)', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            seq.clear()  # Reset sequence so it doesn’t use old frames

        cv2.imshow("ISL Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
