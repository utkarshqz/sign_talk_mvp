import numpy as np
import os

# ======================
# CONFIG
# ======================
TARGET_LENGTH = 30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_DIR = os.path.join(BASE_DIR, "data", "normalized")
FIXED_DIR = os.path.join(BASE_DIR, "data", "fixed")

os.makedirs(FIXED_DIR, exist_ok=True)

def fix_length(sequence, target_len=30):
    T = sequence.shape[0]

    if T == target_len:
        return sequence

    if T < target_len:
        pad_count = target_len - T
        pad_frames = np.repeat(sequence[-1][None, :, :], pad_count, axis=0)
        return np.concatenate([sequence, pad_frames], axis=0)

    indices = np.linspace(0, T - 1, target_len).astype(int)
    return sequence[indices]

for gesture in os.listdir(NORM_DIR):
    norm_gesture_dir = os.path.join(NORM_DIR, gesture)
    fixed_gesture_dir = os.path.join(FIXED_DIR, gesture)

    os.makedirs(fixed_gesture_dir, exist_ok=True)

    for file in os.listdir(norm_gesture_dir):
        if not file.endswith(".npy"):
            continue

        path = os.path.join(norm_gesture_dir, file)
        sequence = np.load(path)

        fixed_seq = fix_length(sequence, TARGET_LENGTH)

        save_path = os.path.join(fixed_gesture_dir, file)
        np.save(save_path, fixed_seq)

        print(f"Fixed length: {gesture}/{file}")

print("All sequences converted to fixed length.")
