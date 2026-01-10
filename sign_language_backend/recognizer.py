import numpy as np
import os

DB_PATH = "reference_db/embeddings.npy"


def recognize(embedding: np.ndarray):
    if not os.path.exists(DB_PATH):
        return "unknown", -1.0

    db = np.load(DB_PATH, allow_pickle=True).item()

    best_label = "unknown"
    best_distance = float("inf")

    for label, ref_embedding in db.items():
        dist = np.linalg.norm(embedding - ref_embedding)
        if dist < best_distance:
            best_distance = dist
            best_label = label

    return best_label, best_distance
