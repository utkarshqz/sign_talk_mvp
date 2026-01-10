from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI()

REFERENCE_DIR = "reference_db"

class EmbeddingInput(BaseModel):
    embedding: list[float]

@app.get("/")
def root():
    return {"status": "Sign Language API running"}

@app.post("/predict")
def predict(data: EmbeddingInput):
    emb = np.array(data.embedding)

    best_label = None
    best_dist = float("inf")

    for file in os.listdir(REFERENCE_DIR):
        if file.endswith(".npy"):
            label = file.replace(".npy", "")
            ref = np.load(os.path.join(REFERENCE_DIR, file))

            dist = np.linalg.norm(ref - emb)
            if dist < best_dist:
                best_dist = dist
                best_label = label

    return {
        "prediction": best_label,
        "distance": float(best_dist)
    }
