import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================
# CONFIG
# =====================================
DATA_DIR = os.path.join(os.getcwd(), "dataset", )  # Path to gesture folders
MODEL_PATH = os.path.join(os.getcwd(), "dataset", "isl_lstm_model.keras")
LABELS_PATH = os.path.join(os.getcwd(), "dataset", "labels.npy")

SEQ_LEN = 50   # Number of frames per gesture sample
FEATURES = 126 # 21 landmarks * 3 coordinates

# =====================================
# LOAD DATA
# =====================================
gestures = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
print("Gestures found:", gestures)

X, y = [], []

for label_idx, gesture in enumerate(gestures):
    folder = os.path.join(DATA_DIR, gesture)
    print(f"Loading {gesture}...")
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(folder, file))
            if seq.shape == (SEQ_LEN, FEATURES):
                X.append(seq)
                y.append(label_idx)
            else:
                print(f"⚠️ Skipped {file} (shape {seq.shape})")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("❌ No valid data found. Please check dataset paths and frame counts.")

print("✅ Data loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================
# SPLIT DATA
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# =====================================
# BUILD MODEL
# =====================================
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(SEQ_LEN, FEATURES)),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================
# EARLY STOPPING
# =====================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# =====================================
# TRAIN MODEL
# =====================================
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# =====================================
# SAVE MODEL & LABELS
# =====================================
print("\n💾 Saving model and labels...")
model.save(MODEL_PATH)
np.save(LABELS_PATH, np.array(gestures))

print("✅ Training complete!")
print(f"✅ Model saved at: {MODEL_PATH}")
print(f"✅ Labels saved at: {LABELS_PATH}")
