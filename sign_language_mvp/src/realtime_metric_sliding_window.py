import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
import time
from collections import deque

# ======================
# CONFIG
# ======================
TARGET_LENGTH = 30          # model was trained on this
EMBEDDING_DIM = 128
THRESHOLD = 0.6

# Sliding window params (NO TIME ASSUMPTION)
WINDOW_SIZE = 30            # frames in rolling window
PREDICT_EVERY = 5           # predict every N frames
SMOOTH_COUNT = 3            # same word N times → accept

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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30 * 42 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM)
        )

    def forward(self, x):
        return self.net(x)

model = EmbeddingNet()
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "embedding_model.pth")))
model.eval()

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ======================
# BUFFERS
# ======================
frame_buffer = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=SMOOTH_COUNT)
stable_words = []
frame_counter = 0

# ======================
# NORMALIZATION
# ======================
WRIST = 0
MIDDLE_MCP = 9

def normalize_sequence(seq):
    norm = []
    for frame in seq:
        frame = np.array(frame)
        ref = frame[WRIST]
        frame = frame - ref
        palm = np.linalg.norm(frame[MIDDLE_MCP])
        if palm > 0:
            frame = frame / palm
        norm.append(frame)
    return np.array(norm)

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

# ======================
# MAIN LOOP
# ======================
print("Pure sliding-window recognition running | Press 'q' to quit")

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

        # pad if only one hand
        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        if len(frame_landmarks) == 42:
            frame_buffer.append(frame_landmarks)
            frame_counter += 1

    # ======================
    # PREDICTION (SLIDING WINDOW)
    # ======================
    if len(frame_buffer) == WINDOW_SIZE and frame_counter % PREDICT_EVERY == 0:
        seq = np.array(frame_buffer)
        seq = normalize_sequence(seq)
        seq = seq.reshape(1, -1)

        with torch.no_grad():
            emb = model(torch.tensor(seq, dtype=torch.float32)).numpy()[0]

        best_gesture = "UNKNOWN"
        best_dist = float("inf")

        for gesture, refs in reference_db.items():
            dists = [cosine_distance(emb, ref) for ref in refs]
            median_dist = np.median(dists)

            if median_dist < best_dist:
                best_dist = median_dist
                best_gesture = gesture

        if best_dist < THRESHOLD:
            prediction_buffer.append(best_gesture)
        else:
            prediction_buffer.append("UNKNOWN")

        # ======================
        # SMOOTHING
        # ======================
        if len(prediction_buffer) == SMOOTH_COUNT:
            if len(set(prediction_buffer)) == 1:
                word = prediction_buffer[0]
                if word != "UNKNOWN":
                    if not stable_words or stable_words[-1] != word:
                        stable_words.append(word)
                        print("Detected:", word)

    # ======================
    # UI
    # ======================
    cv2.putText(
        frame,
        "Words: " + " | ".join(stable_words[-6:]),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("Sliding Window Sign Recognition", frame)
    time.sleep(0.03)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#not working that much properly because of no time assumption