import cv2
import os
import mediapipe as mp
import numpy as np
import time

# ---------- CONFIG ----------
DATA_DIR = os.path.join(os.getcwd(), 'dataset', 'dataset')
SEQ_LEN = 50
SAMPLES_PER_LABEL = 50
CAM_INDEX = 0

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Create Base Folder ----------
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Function: Collect Data ----------
def collect_data_for_label(label_name):
    safe_label = label_name.replace(" ", "_")
    folder_path = os.path.join(DATA_DIR, safe_label)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        try:
            print(f"\n=== Collecting data for: {safe_label} ===")
            input("Press ENTER when ready to start...")

            existing_files = len(os.listdir(folder_path))
            for sample_idx in range(existing_files, existing_files + SAMPLES_PER_LABEL):
                seq = []
                print(f"\nRecording sample {sample_idx - existing_files + 1}/{SAMPLES_PER_LABEL}")

                # Countdown before recording
                for t in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f'Starting in {t}', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.imshow('Collect', frame)
                    cv2.waitKey(1000)

                print("▶ Recording started...")

                start_time = time.time()
                while len(seq) < SEQ_LEN:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    row = []
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for lm in hand_landmarks.landmark:
                                row.extend([lm.x, lm.y, lm.z])
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        # No hand detected
                        row = [0.0] * 126

                    # Make sure row always has 126 values
                    row = (row + [0.0] * 126)[:126]
                    seq.append(row)

                    cv2.putText(frame, f'{safe_label}: Frame {len(seq)}/{SEQ_LEN}', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Collect', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                duration = time.time() - start_time
                print(f"✅ Captured {SEQ_LEN} frames in {duration:.2f}s")

                filename = f'{safe_label}_{sample_idx}.npy'
                np.save(os.path.join(folder_path, filename), np.array(seq))
                print(f"💾 Saved: {filename}")

        except KeyboardInterrupt:
            print("\n⚠️ Recording stopped manually.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Finished recording for {safe_label}.\n")

# ---------- Main Loop ----------
while True:
    print("\nAvailable gestures:")
    gestures = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if gestures:
        for i, g in enumerate(gestures):
            print(f"{i+1}. {g}")
    else:
        print("(None yet)")

    print("\nOptions:")
    print(" - Enter the number of an existing gesture to add more samples")
    print(" - OR type a new gesture name to create it")
    print(" - OR type 'exit' to quit")

    choice = input("\nYour choice: ").strip()

    if choice.lower() == 'exit':
        print("👋 Exiting data collection.")
        break
    elif choice.isdigit() and 1 <= int(choice) <= len(gestures):
        label_name = gestures[int(choice) - 1]
    else:
        label_name = choice

    collect_data_for_label(label_name)
