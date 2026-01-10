import numpy as np
import os

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
NORM_DIR = os.path.join(BASE_DIR, "data", "normalized")

os.makedirs(NORM_DIR, exist_ok=True)

# ======================
# LANDMARK INDICES
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
