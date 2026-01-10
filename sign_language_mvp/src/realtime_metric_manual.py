import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time

# ======================
# CONFIG
# ======================
TARGET_LENGTH = 30
EMBEDDING_DIM = 128
THRESHOLD = 0.6   # tune later

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REF_DIR = os.path.join(BASE_DIR, "reference_db")

# ======================
# LOAD REFERENCE DB
# ======================
reference_db = {}
for file in os.listdir(REF_DIR):
    if file.endswith(".npy"):
        gesture = file.replace(".npy", "")
        reference_db[gesture] = np.load(os.path.join(REF_DIR, file))

print("Loaded reference gestures:", list(reference_db.keys()))

# ======================
# EMBEDDING MODEL
# ======================
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )

    def forward(self, x):
        return self.net(x)

# 🔥 UPDATED INPUT SIZE: 30 * 42 * 2
model = EmbeddingNet(30 * 42 * 2, EMBEDDING_DIM)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "embedding_model.pth")))
model.eval()

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,   # 🔥 TWO HANDS ENABLED
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

recording = False
sequence = []
prediction = ""

print("Press 's' to START/STOP recording | 'q' to quit")

# ======================
# NORMALIZATION
# ======================
WRIST = 0
MIDDLE_MCP = 9

def normalize_sequence(seq):
    norm = []
    for frame in seq:
        frame = np.array(frame)

        # Reference: wrist of first hand
        ref = frame[WRIST]
        frame = frame - ref

        palm = np.linalg.norm(frame[MIDDLE_MCP])
        if palm > 0:
            frame = frame / palm

        norm.append(frame)
    return np.array(norm)

def fix_length(seq, target=30):
    T = seq.shape[0]
    if T < target:
        pad = np.repeat(seq[-1][None, :, :], target - T, axis=0)
        return np.concatenate([seq, pad])
    idx = np.linspace(0, T - 1, target).astype(int)
    return seq[idx]

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

# ======================
# MAIN LOOP
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        frame_landmarks = []

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_points = [[p.x, p.y] for p in hand_landmarks.landmark]
            frame_landmarks.extend(hand_points)

        # Pad if only one hand
        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        if recording and len(frame_landmarks) == 42:
            sequence.append(frame_landmarks)

    if key == ord('s'):
        if not recording:
            recording = True
            sequence = []
            prediction = ""
            print("Recording started")
        else:
            recording = False
            print("Recording stopped")

            if len(sequence) > 5:
                seq = np.array(sequence)
                seq = normalize_sequence(seq)
                seq = fix_length(seq, TARGET_LENGTH)
                seq = seq.reshape(1, -1)

                with torch.no_grad():
                    emb = model(torch.tensor(seq, dtype=torch.float32)).numpy()[0]

                best_gesture = "UNKNOWN"
                best_dist = float("inf")

                for gesture, refs in reference_db.items():
                    dists = [cosine_distance(emb, ref) for ref in refs]
                    mean_dist = np.mean(dists)

                    if mean_dist < best_dist:
                        best_dist = mean_dist
                        best_gesture = gesture

                prediction = best_gesture if best_dist < THRESHOLD else "UNKNOWN"

                print("Prediction:", prediction, "| Distance:", round(best_dist, 3))

    if key == ord('q'):
        break

    status = "RECORDING" if recording else "IDLE"
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if recording else (0, 255, 0), 2)

    cv2.putText(frame, f"Prediction: {prediction}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Metric Gesture Recognition", frame)
    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
#working good , but perform "thanks" sign slightly tilted