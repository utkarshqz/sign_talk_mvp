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
THRESHOLD = 0.6

# Continuous recognition tuning
MOTION_THRESHOLD = 0.01
PAUSE_FRAMES = 10
MIN_SIGN_FRAMES = 18
MIN_TOTAL_MOTION = 0.5

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
current_sequence = []
previous_frame = None
pause_counter = 0
total_motion = 0.0
word_buffer = []

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

def motion_score(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.mean(np.linalg.norm(f2 - f1, axis=1))

# ======================
# MAIN LOOP
# ======================
print("Continuous sign recognition running | Press 'q' to quit")

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

        # pad if one hand
        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        if len(frame_landmarks) == 42:
            if previous_frame is not None:
                motion = motion_score(previous_frame, frame_landmarks)

                if motion > MOTION_THRESHOLD:
                    current_sequence.append(frame_landmarks)
                    total_motion += motion
                    pause_counter = 0
                else:
                    pause_counter += 1

                # SIGN END DETECTED
                if (
                    pause_counter >= PAUSE_FRAMES
                    and len(current_sequence) >= MIN_SIGN_FRAMES
                    and total_motion >= MIN_TOTAL_MOTION
                ):
                    seq = np.array(current_sequence)
                    seq = normalize_sequence(seq)
                    seq = fix_length(seq, TARGET_LENGTH)
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
                        word_buffer.append(best_gesture)
                        print("Detected:", best_gesture, "| Distance:", round(best_dist, 3))

                    # RESET FOR NEXT SIGN
                    current_sequence = []
                    pause_counter = 0
                    total_motion = 0.0

            previous_frame = frame_landmarks

    if key == ord('q'):
        break

    # ======================
    # UI
    # ======================
    cv2.putText(
        frame,
        "Words: " + " | ".join(word_buffer[-6:]),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("Continuous Sign Recognition", frame)
    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
