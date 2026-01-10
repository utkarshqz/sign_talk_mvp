import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ======================
# CONFIG
# ======================
GESTURE_NAME = "good"     # 🔁 change this per gesture
SAVE_DIR = os.path.join("data", "raw", GESTURE_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,        # supports 1-hand & 2-hand signs
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ======================
# STATE VARIABLES
# ======================
sequence = []
recording = False
sample_id = len(os.listdir(SAVE_DIR))
last_saved = sample_id

print("Press 's' to START/STOP recording | 'q' to quit")

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

    # ======================
    # LANDMARK CAPTURE
    # ======================
    if result.multi_hand_landmarks:
        frame_landmarks = []

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            hand_points = [[p.x, p.y] for p in hand_landmarks.landmark]
            frame_landmarks.extend(hand_points)

        # pad if only one hand
        if len(frame_landmarks) == 21:
            frame_landmarks.extend([[0.0, 0.0]] * 21)

        if recording and len(frame_landmarks) == 42:
            sequence.append(frame_landmarks)

    # ======================
    # KEY EVENTS
    # ======================
    if key == ord('s'):
        if not recording:
            recording = True
            sequence = []
            print("▶ Recording started")
        else:
            recording = False
            print("⏹ Recording stopped")

            if len(sequence) > 10:
                save_path = os.path.join(
                    SAVE_DIR, f"{GESTURE_NAME}_{sample_id}.npy"
                )
                np.save(save_path, np.array(sequence))
                sample_id += 1
                last_saved = sample_id
                print(f"✔ Saved sample {sample_id}")

    if key == ord('q'):
        break

    # ======================
    # UI OVERLAY
    # ======================
    status_text = "RECORDING" if recording else "IDLE"
    status_color = (0, 0, 255) if recording else (0, 255, 0)

    cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"Status: {status_text}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    cv2.putText(frame, f"Samples recorded: {sample_id}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Capture Landmarks", frame)
    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()
