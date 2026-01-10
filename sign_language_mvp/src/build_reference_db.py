import os
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fixed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REF_DIR = os.path.join(BASE_DIR, "reference_db")
os.makedirs(REF_DIR, exist_ok=True)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30 * 42 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.net(x)

model = EmbeddingNet()
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "embedding_model.pth")))
model.eval()

with torch.no_grad():
    for gesture in os.listdir(DATA_DIR):
        embeddings = []
        for f in os.listdir(os.path.join(DATA_DIR, gesture)):
            seq = np.load(os.path.join(DATA_DIR, gesture, f))
            seq = torch.tensor(seq.reshape(1, -1), dtype=torch.float32)
            emb = model(seq).numpy()[0]
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        np.save(os.path.join(REF_DIR, f"{gesture}.npy"), embeddings)
        print(f"{gesture}: {embeddings.shape}")

print("Reference DB built.")
