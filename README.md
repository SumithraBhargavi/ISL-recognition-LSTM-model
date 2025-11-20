# Indian Sign Language (ISL) Gesture Recognition – LSTM + MediaPipe Hands

This project implements a **real-time Indian Sign Language (ISL) gesture recognition system** using:

- **MediaPipe Hands** for extracting 3D hand landmarks  
- **LSTM deep learning model** for gesture sequence classification  
- **OpenCV** for real-time webcam-based inference  

The system supports collecting gesture data, training an LSTM model, and running live gesture detection.

---

## 🚀 Features

### ✔ Data Collection  
- Capture hand gesture sequences using your webcam  
- Extract 21×3 hand landmarks for up to **2 hands** (126 features per frame)  
- Save each gesture as `.npy` sequences  
- 30 frames per sequence and 30 samples per gesture  

### ✔ LSTM Model Training  
- Train on collected `.npy` sequences  
- Uses a 2-layer LSTM architecture  
- Automatically splits data into train/test sets  
- Saves:  
  - Trained model → `isl_lstm_model.keras`  
  - Label file → `labels.npy`

### ✔ Real-Time Gesture Detection  
- Detects hand landmarks from webcam  
- Maintains a rolling sequence of 50 frames  
- Predicts gesture using the trained LSTM model  
- Displays gesture name + confidence %

---

## 📁 Project Structure
├── dataset/
│ ├── gesture1/
│ ├── gesture2/
│ └── ...
├── data_collection.py
├── train_model.py
├── gesture_detection.py
├── labels.txt
└── README.md


