import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import requests
import time

# =========================
# CONFIG
# =========================
API_URL = "https://sign-language-api-utkar-g6csdrhweadcgfft.centralindia-01.azurewebsites.net/predict"
FRAME_SKIP = 5

st.set_page_config(page_title="Sign Language Recognition", layout="centered")
st.title("🤟 Live Sign Language Recognition (Two-Hand)")
st.caption("Normalized landmarks + Azure backend")

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# UI
# =========================
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])
prediction_box = st.empty()

cap = cv2.VideoCapture(0)

frame_count = 0
last_prediction = "—"

# =========================
# NORMALIZATION
# =========================
def normalize_hand(hand_landmarks):
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

    wrist = pts[0]
    pts = pts - wrist  # translation invariance

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale  # scale invariance

    return pts.flatten()  # shape (42,)

def build_embedding(multi_hand_landmarks):
    emb = []

    for hand in multi_hand_landmarks:
        emb.extend(normalize_hand(hand))

    # If only one hand, pad second hand
    if len(multi_hand_landmarks) == 1:
        emb.extend([0.0] * 42)

    # Pad / trim to backend size (128)
    if len(emb) < 128:
        emb.extend([0.0] * (128 - len(emb)))
    else:
        emb = emb[:128]

    return emb

# =========================
# MAIN LOOP
# =========================
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for h in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            embedding = build_embedding(result.multi_hand_landmarks)

            try:
                response = requests.post(
                    API_URL,
                    json={"embedding": embedding},
                    timeout=3
                )
                if response.status_code == 200:
                    last_prediction = response.json().get("prediction", "unknown")
            except:
                last_prediction = "API error"

    prediction_box.markdown(
        f"### 🟢 Prediction: **{last_prediction.upper()}**"
    )

    FRAME_WINDOW.image(frame)
    time.sleep(0.02)

cap.release()
