import os
import numpy as np

# ======================
# CONFIG
# ======================
TARGET_LENGTH = 30

RAW_DIR = os.path.join("data", "raw")
FIXED_DIR = os.path.join("data", "fixed")

os.makedirs(FIXED_DIR, exist_ok=True)

# ======================
# NORMALIZATION
# ======================
WRIST = 0
MIDDLE_MCP = 9

def normalize_sequence(seq):
    """
    seq shape: (T, 42, 2)
    """
    norm = []

    for frame in seq:
        frame = np.array(frame)

        # Reference: wrist of first hand
        ref = frame[WRIST]
        frame = frame - ref

        # Scale using palm size (first hand)
        palm = np.linalg.norm(frame[MIDDLE_MCP])
        if palm > 0:
            frame = frame / palm

        norm.append(frame)

    return np.array(norm)

# ======================
# FIX LENGTH
# ======================
def fix_length(seq, target_len=30):
    T = seq.shape[0]

    if T < target_len:
        pad = np.repeat(seq[-1][None, :, :], target_len - T, axis=0)
        return np.concatenate([seq, pad])

    idx = np.linspace(0, T - 1, target_len).astype(int)
    return seq[idx]

# ======================
# PROCESS ALL GESTURES
# ======================
for gesture in os.listdir(RAW_DIR):
    raw_gesture_dir = os.path.join(RAW_DIR, gesture)
    fixed_gesture_dir = os.path.join(FIXED_DIR, gesture)

    os.makedirs(fixed_gesture_dir, exist_ok=True)

    print(f"\nProcessing gesture: {gesture}")

    for file in os.listdir(raw_gesture_dir):
        if not file.endswith(".npy"):
            continue

        path = os.path.join(raw_gesture_dir, file)
        seq = np.load(path)   # (T, 42, 2)

        if seq.shape[1:] != (42, 2):
            print(f"⚠️ Skipping invalid file: {file} | shape: {seq.shape}")
            continue

        seq = normalize_sequence(seq)
        seq = fix_length(seq, TARGET_LENGTH)

        save_path = os.path.join(fixed_gesture_dir, file)
        np.save(save_path, seq)

    print(f"✔ Saved fixed samples to: {fixed_gesture_dir}")

print("\n✅ Normalization + fix-length complete.")

