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
CAPTURE_SECONDS = 2.0
TARGET_LENGTH = 30
EMBEDDING_DIM = 128
DEFAULT_THRESHOLD = 0.6

# Per-class thresholds (CRITICAL FIX)
CLASS_THRESHOLDS = {
    "me": 0.45,        # stricter
    "thanks": 0.65,    # more lenient
    "hello": 0.6,
    "yes": 0.6,
    "good": 0.6
}

# Motion sanity (CRITICAL FIX)
ME_MOTION_REJECT_THRESHOLD = 0.015

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
# NORMALIZATION & UTILS
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

def total_motion(seq):
    seq = np.array(seq)
    diffs = np.diff(seq, axis=0)
    return np.mean(np.linalg.norm(diffs, axis=2))

# ======================
# STATE
# ======================
capturing = False
sequence = []
start_time = None
prediction = ""
sentence_words = []

print("2-second sentence mode (FIXED) | 'c' clear | 'q' quit")

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
    now = time.time()

    if result.multi_hand_landmarks:
        frame_landmarks = []

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_points = [[p.x, p.y] for p in hand_landmarks.landmark]
            frame_landmarks.extend(hand_points)

        # pad single hand
        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        # auto start capture
        if not capturing and len(frame_landmarks) == 42:
            capturing = True
            start_time = now
            sequence = []
            prediction = ""
            print("Capture started")

        if capturing and len(frame_landmarks) == 42:
            sequence.append(frame_landmarks)

    # ======================
    # END CAPTURE AFTER 2 SEC
    # ======================
    if capturing and (now - start_time) >= CAPTURE_SECONDS:
        capturing = False
        print("Capture ended")

        if len(sequence) >= 10:
            seq = normalize_sequence(sequence)
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

            threshold = CLASS_THRESHOLDS.get(best_gesture, DEFAULT_THRESHOLD)
            motion = total_motion(sequence)

            if best_dist < threshold:
                # motion sanity: reject "me" if motion is high
                if best_gesture == "me" and motion > ME_MOTION_REJECT_THRESHOLD:
                    prediction = "UNKNOWN"
                    print("Rejected 'me' (motion too high)")
                else:
                    prediction = best_gesture
                    if not sentence_words or sentence_words[-1] != prediction:
                        sentence_words.append(prediction)
                        print("Detected:", prediction,
                              "| dist:", round(best_dist, 3),
                              "| motion:", round(motion, 4))
            else:
                prediction = "UNKNOWN"

        sequence = []

    # ======================
    # UI
    # ======================
    status = "CAPTURING..." if capturing else "READY"
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if capturing else (0, 255, 0), 2)

    cv2.putText(frame, f"Last word: {prediction}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(
        frame,
        "Sentence: " + " ".join(sentence_words[-8:]),
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("2-Second Sentence Recognition (Fixed)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        sentence_words = []
        print("Sentence cleared")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# running smoothly if perform signs slowly :)