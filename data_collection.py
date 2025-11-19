import cv2
import os
import time
import mediapipe as mp
import numpy as np

# ---------- CONFIG ----------
LABELS_FILE = 'labels.txt'
DATA_DIR = os.path.join(os.getcwd(), 'dataset')  # dataset folder in project
SEQ_LEN = 30               # frames per sample
SAMPLES_PER_LABEL = 30     # number of sequences per label
CAM_INDEX = 0              # webcam index

# ---------- Mediapipe ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Load all labels ----------
with open(LABELS_FILE, 'r', encoding='utf-8-sig') as f:
    all_labels = [l.strip() for l in f.readlines() if l.strip()]

# ---------- Select gesture to collect ----------
print("Available gestures to collect:")
for i, lbl in enumerate(all_labels):
    print(f"{i+1}. {lbl}")

choice = input("Enter the gesture name you want to collect now: ").strip()

if choice not in all_labels:
    print(f"'{choice}' is not in labels.txt. Exiting...")
    exit()

labels = [choice]  # only collect the selected gesture
print(f"Selected gesture: {labels[0]}")

# ---------- Setup webcam ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    try:
        for lbl in labels:
            safe_label = lbl.replace(" ", "_")
            folder_path = os.path.join(DATA_DIR, safe_label)
            os.makedirs(folder_path, exist_ok=True)  # ensure folder exists

            print(f"\n=== Collecting for: {lbl} ===")
            input("Press ENTER when ready to start recording this gesture...")

            collected = 0
            sample_idx = len(os.listdir(folder_path))

            while collected < SAMPLES_PER_LABEL:
                seq = []
                print(f"Recording sample {collected+1}/{SAMPLES_PER_LABEL}")

                # --- Countdown before recording ---
                for t in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f'Get ready: {t}', (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow('Collect', frame)
                    cv2.waitKey(1000)

                # --- Capture sequence ---
                while len(seq) < SEQ_LEN:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    row = []

                    # --- Extract landmarks ---
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for lm in hand_landmarks.landmark:
                                row.extend([lm.x, lm.y, lm.z])
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Ensure row always has 126 elements
                    row += [0.0] * (126 - len(row))  # fill missing values

                    # Append only valid rows
                    if len(row) == 126:
                        seq.append(row)
                    else:
                        print(f"Skipped frame, incorrect row length: {len(row)}")

                    # --- Display progress ---
                    cv2.putText(frame, f'{lbl} {len(seq)}/{SEQ_LEN}', (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow('Collect', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                # --- Save sequence ---
                filename = f'{safe_label}_{sample_idx}.npy'
                np.save(os.path.join(folder_path, filename), np.array(seq))
                print(f"Saved sample {collected+1}/{SAMPLES_PER_LABEL} as {filename}")

                sample_idx += 1
                collected += 1

    except KeyboardInterrupt:
        print("\nData collection stopped manually.")

cap.release()
cv2.destroyAllWindows()
print("✅ Data collection complete!")
