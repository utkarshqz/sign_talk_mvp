import cv2
import mediapipe as mp
import numpy as np
import os
import imageio

MEDIA_DIR = os.path.join("data", "media")
RAW_DIR = os.path.join("data", "raw")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def process_frames(frames, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    sample_id = len(os.listdir(save_dir))
    sequence = []

    for frame in frames:
        if frame is None:
            continue

        if len(frame.shape) == 4:  # GIF frame
            frame = frame[:, :, :3]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            continue

        frame_landmarks = []

        for hand_landmarks in result.multi_hand_landmarks:
            hand_points = [[p.x, p.y] for p in hand_landmarks.landmark]
            frame_landmarks.extend(hand_points)

        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        if len(frame_landmarks) == 42:
            sequence.append(frame_landmarks)

    if len(sequence) < 5:
        print(f"⚠️ Skipped {prefix} (too few landmarks)")
        return

    path = os.path.join(save_dir, f"{prefix}_{sample_id}.npy")
    np.save(path, np.array(sequence))
    print(f"✔ Saved {path}")

def read_media(path):
    ext = path.lower()

    if ext.endswith((".gif", ".webp")):
        frames = imageio.mimread(path)
        return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# ======================
# MAIN PROCESS
# ======================
for gesture in os.listdir(MEDIA_DIR):
    raw_gesture_dir = os.path.join(RAW_DIR, gesture)

    # 🔒 skip gestures already captured
    if os.path.exists(raw_gesture_dir) and len(os.listdir(raw_gesture_dir)) > 0:
        print(f"⏭ Skipping {gesture} (already exists)")
        continue

    media_gesture_dir = os.path.join(MEDIA_DIR, gesture)
    os.makedirs(raw_gesture_dir, exist_ok=True)

    print(f"\nProcessing gesture: {gesture}")

    for file in os.listdir(media_gesture_dir):
        media_path = os.path.join(media_gesture_dir, file)
        frames = read_media(media_path)
        process_frames(frames, raw_gesture_dir, gesture)

print("\n✅ Media processing complete.")
